#include "pch.h"

#include <nrbridge/NrBridge.h>
#include <nrbridge/NrPipeline.h>

#include "Config.h"
#include "State.h"
#include "Util.h"

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// The command-list split, taken from the reference integration without modification. It is
// included after everything of the host's, because it brings its own Detours header.
#include "submission/CommandListProxy.h"
#include "submission/SubmissionHooks.h"

// The reference's evaluate-time cut, likewise verbatim: TrySplitAtEvaluate, the between slot
// and the pending enqueue it consumes. It is the piece that turns "the list may be cut" into
// a cut, and it is included rather than rewritten for the reason given in NrBridge.h.
//
// It sits one directory below where it is used, because that is where it sat in the reference:
// its own includes are "Selector.h" beside it and "../submission/CommandListProxy.h" above it,
// so it needs a sibling of submission/, not a child.
#include "cut/LmxxfEvaluateCut.h"

// The Vulkan side of the hand-over: the device on the game's adapter, and the two shared fences
// the slot's submission waits on and signals. Separate module, for the same reason the reference
// keeps its pe bridge separate.
#include "VulkanGap.h"

// How the frame's pixels get across: a linear buffer, shared by handle, that both APIs copy a
// texture's rows into and out of. The reference's shape, not a shared texture, and its sizes
// are the frame's own.
#include "SharedBuffer.h"

// The shape of everything here comes from the reference integration; see NrBridge.h. What is
// ours is only the part that has to be: reading this host's parameter block, and resolving the
// state each resource is actually in before anything is recorded.
namespace
{
// Reading the parameter block is deliberately done the way the rest of the host does it: a typed
// Get first, then the untyped one. The host sets these keys from two different places (the FFX
// path and the NGX path) and they do not agree on which overload is populated, so a single Get
// would silently read nothing on one of them.
ID3D12Resource* ReadResource(NVSDK_NGX_Parameter* p, const char* key)
{
    ID3D12Resource* r = nullptr;
    if (p->Get(key, &r) != NVSDK_NGX_Result_Success)
        p->Get(key, reinterpret_cast<void**>(&r));
    return r;
}

using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point from, Clock::time_point to)
{
    return std::chrono::duration<double, std::milli>(to - from).count();
}

float ReadFloat(NVSDK_NGX_Parameter* p, const char* key, float fallback)
{
    float v = fallback;
    if (p->Get(key, &v) != NVSDK_NGX_Result_Success)
        return fallback;
    return v;
}

unsigned int ReadUInt(NVSDK_NGX_Parameter* p, const char* key)
{
    unsigned int v = 0;
    if (p->Get(key, &v) != NVSDK_NGX_Result_Success)
        return 0;
    return v;
}

// Whether this game is on Unreal, which decides what state several of these resources are left
// in. The test is the host's own, copied from the XeSS path that has to ask the same question.
bool IsUnreal()
{
    // The third test is written the way the host writes it: gameQuirks is a flag_set, which
    // answers a truth question directly and does not compare against an integer.
    return State::Instance().NVNGX_Engine == NVSDK_NGX_ENGINE_TYPE_UNREAL ||
           State::Instance().GameEngine == GameEngineType::Unreal ||
           State::Instance().gameQuirks & GameQuirk::ForceUnrealEngine;
}

D3D12_RESOURCE_STATES ResolveState(const CustomOptional<int32_t, NoDefault>& configured,
                                   D3D12_RESOURCE_STATES unrealFallback, D3D12_RESOURCE_STATES otherFallback)
{
    // Configured wins: the user set it because this game is the exception. Otherwise the answer
    // is what the host's own upscaler assumes, which differs between Unreal and everything else
    // -- and a wrong StateBefore is not an error that fails politely on a shipped driver.
    if (configured.has_value())
        return static_cast<D3D12_RESOURCE_STATES>(configured.value());

    return IsUnreal() ? unrealFallback : otherFallback;
}

const char* FormatName(DXGI_FORMAT f)
{
    switch (f)
    {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
        return "R8G8B8A8_UNORM";
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
        return "R16G16B16A16_FLOAT";
    case DXGI_FORMAT_R10G10B10A2_UNORM:
        return "R10G10B10A2_UNORM";
    case DXGI_FORMAT_R11G11B10_FLOAT:
        return "R11G11B10_FLOAT";
    case DXGI_FORMAT_R16G16_FLOAT:
        return "R16G16_FLOAT";
    case DXGI_FORMAT_R32G8X24_TYPELESS:
        return "R32G8X24_TYPELESS";
    default:
        return "?";
    }
}

// One transition, all subresources, and nothing at all when the state asked for is the state
// the resource is already in -- which is the usual case for the buffers, and the case that has
// to cost no barrier.
void Transition(ID3D12GraphicsCommandList* c, ID3D12Resource* r, D3D12_RESOURCE_STATES before,
                D3D12_RESOURCE_STATES after)
{
    if (r == nullptr || before == after)
        return;

    D3D12_RESOURCE_BARRIER b {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    c->ResourceBarrier(1, &b);
}

// The reference's CopyToBuffer, and the way back out of the buffer into the texture the
// upscaler will read. Both take the texture into the copy's own state and hand it back in the
// state it was found in: that state belongs to the game, and the upscaler expects to find it.
void CopyToBuffer(ID3D12GraphicsCommandList* c, ID3D12Resource* texture, D3D12_RESOURCE_STATES state,
                  ID3D12Resource* buffer, const NrBridge::Footprint& fp, UINT w, UINT h)
{
    Transition(c, texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    Transition(c, buffer, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);

    D3D12_TEXTURE_COPY_LOCATION dst { buffer, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {} };
    dst.PlacedFootprint = fp.d3d;
    D3D12_TEXTURE_COPY_LOCATION src { texture, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {} };
    const D3D12_BOX box { 0, 0, 0, w, h, 1 };
    c->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);

    Transition(c, buffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    Transition(c, texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
}

void CopyFromBuffer(ID3D12GraphicsCommandList* c, ID3D12Resource* texture, ID3D12Resource* buffer,
                    const NrBridge::Footprint& fp, UINT w, UINT h, D3D12_RESOURCE_STATES state,
                    D3D12_RESOURCE_STATES textureNow)
{
    Transition(c, buffer, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    Transition(c, texture, textureNow, D3D12_RESOURCE_STATE_COPY_DEST);

    D3D12_TEXTURE_COPY_LOCATION dst { texture, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {} };
    D3D12_TEXTURE_COPY_LOCATION src { buffer, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {} };
    src.PlacedFootprint = fp.d3d;
    c->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    Transition(c, texture, D3D12_RESOURCE_STATE_COPY_DEST, state);
    Transition(c, buffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
}

// The reference's ResultDesc: the upscaler's colour texture, copied. One deliberate deviation
// from the reference, and the reason is the host, not taste: the reference hands its result to
// an upscaler that reads it as a shader resource and never takes the texture through the
// render-target state, so it drops every flag. This host's upscaler is OptiScaler, whose
// Unreal path assumes the colour it is handed sits in the render-target state and barriers it
// from there (XeSSFeature_Dx12.cpp: ColorResourceBarrier = RENDER_TARGET). A texture without
// ALLOW_RENDER_TARGET may not legally enter that state at all -- the runtime refuses the
// barrier when the list closes, with E_INVALIDARG, and its debug layer, by its own
// documentation, need not say a word about it. That was the crash of 2026-10-13.
D3D12_RESOURCE_DESC ResultDesc(const D3D12_RESOURCE_DESC& colour, UINT w, UINT h)
{
    D3D12_RESOURCE_DESC rd = colour;
    rd.Width = w;
    rd.Height = h;
    rd.MipLevels = 1;
    rd.DepthOrArraySize = 1;
    rd.SampleDesc = { 1, 0 };
    rd.Alignment = 0;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    return rd;
}

std::string Describe(const char* what, ID3D12Resource* r)
{
    if (r == nullptr)
        return std::string(what) + "=absent";

    auto d = r->GetDesc();
    char buf[192];
    std::snprintf(buf, sizeof(buf), "%s=%ux%u %s(%u)%s%s", what, static_cast<unsigned>(d.Width),
                  static_cast<unsigned>(d.Height), FormatName(d.Format), static_cast<unsigned>(d.Format),
                  (d.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) ? " rt" : "",
                  (d.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) ? " uav" : "");
    return std::string(buf);
}

std::atomic<unsigned long long> g_seen { 0 };
std::atomic<bool> g_contractLogged { false };
std::atomic<bool> g_failed { false };
std::atomic<bool> g_hooksTried { false };
std::atomic<NrBridge::Backend*> g_backend { nullptr };
std::mutex g_backendMutex;
std::string g_failureReason;

// What the cut answered, counted as the reference's own return value means it: S_OK is a cut,
// S_FALSE is "not a list we wrapped" (its documented meaning, not an error), anything else is
// a refusal with a reason of its own.
std::atomic<unsigned long long> g_splitOk { 0 };
std::atomic<unsigned long long> g_splitNoProxy { 0 };
std::atomic<unsigned long long> g_splitFailed { 0 };
// `g_splitNoProxy` is one counter over four different reasons, and which one dominates decides
// where the work is: if it is mostly the list being the game's own, the hook's timing is the
// problem; if it is the proxy refusing its own list, no amount of earlier hooking helps and the
// cut has to become possible instead. Counting them apart is the difference between those two
// searches, and it costs four atomic increments on a path that already does a COM query.
std::atomic<unsigned long long> g_declNotOurs { 0 };
std::atomic<unsigned long long> g_declIneligible { 0 };
std::atomic<unsigned long long> g_declNoResources { 0 };
std::atomic<unsigned long long> g_declNoCompute { 0 };
// The HRESULT of the last cut, kept only so it can be printed the way the reference prints it:
// its own boundary line carries cutHr={:X}, because a cut that stopped working shows up there
// before it shows up anywhere else.
std::atomic<HRESULT> g_lastCutHr { S_OK };

// The split can only be made on a list this host wrapped as it was created, so the wrapping
// hook has to be in place before the game asks for another one. Nothing is cut here: this only
// reports whether the list the game handed us is one that could be.
void EnsureSubmissionArmed(ID3D12Device* device, ID3D12CommandQueue* queue)
{
    if (g_hooksTried.exchange(true, std::memory_order_acq_rel))
        return;

    if (!Config::Instance()->NrWrapLists.value_or_default())
    {
        LOG_INFO("NrBridge: WrapLists is off -- command lists are left as the game made them");
        return;
    }

    /* Wrap the lists the game made before this point, and the reference says why in so
     * many words beside this switch it owns: "UE can retain lists created before the first
     * swapchain. Wrap only calls from the host executable during boot; Streamline and vendor
     * modules stay native." (`SubmissionHooks.h`, above g_earlyExeWrap).
     *
     * This is that game. The log says what it costs: `declined=1813` against
     * `cutsRecorded=586`, every decline reading "(not our list)" -- three evaluates out of
     * four arrive with a command list the proxy never wrapped, so three frames out of four
     * are shown the game's own colour, which is the flicker. The hook is armed here, once,
     * on the swapchain's queue; lists created before it are still the game's own and no
     * later switch reaches them.
     *
     * The boot lists that were already made are still the game's own: that is what
     * `ArmSubmissionAtDeviceCreation` exists for, and it ran before this frame. What is
     * missing here is only the game's own queue, so this switch is not touched -- the
     * reference decides early wrapping once, at the device, against its whitelist. */
    DlssNr::Submission::Hooks::SetProxyWrap(true);
    DlssNr::Submission::Hooks::SetWrapOpenLists(true);
    const auto hr = DlssNr::Submission::Hooks::Arm(device, queue);
    LOG_INFO("NrBridge: submission hooks armed 0x{:x} on queue {}", static_cast<unsigned>(hr),
             static_cast<void*>(queue));

    // The between slot is where the reference's own renderer is invoked, in the gap between the
    // two halves of a cut list. It is armed here rather than at the cut: the reference does the
    // same (ArmBetweenSlot installs BetweenThunk), and with nothing enqueued the thunk finds an
    // empty pending and returns, which is what makes the cut itself measurable on its own.
    if (SUCCEEDED(hr))
        DlssNr::Backend::LmxxfCut::ArmBetweenSlot();
}

// The gap between the two halves of a cut list. This is where the reference runs its renderer:
// its backend records the network's inputs, cuts the list, and the runtime's CoreEnqueueHip is
// called in the gap, before the outputs are recorded and the upscaler is handed the result.
//
// No renderer is attached yet, so nothing has ever been enqueued and the slot has never had
// anything to do -- which is why the previous step could not tell whether it ran. Its counters
// read zero either way, because the reference's BetweenThunk returns before it counts anything
// when the pending is empty.
//
// What goes in now is a probe, and it is the reference's own shape, not a smaller version of a
// different one. Its between work never executes a command list in the slot -- its comment is
// verbatim "an ExecuteCommandLists from here would re-enter the game's detoured one" -- it
// signals a D3D12 fence on the game's queue and gives the work to the Vulkan queue. The signal
// alone is the least that keeps that shape: a queue command that touches no resource, knows no
// resource state, and whose value a CPU thread can read. Two earlier attempts submitted a
// command list of our own here and both removed the device on the first slot run, which is the
// reference's warning taken as fact rather than as advice. The question the probe answers is
// still the same -- does the slot run, and what does standing in it cost.
constexpr UINT64 GapFenceBase = 1;

// Stands in for the reference's job handle until there is a job. Its own handles are odd, so
// never null and never a pointer, and BetweenThunk refuses to call in with a null session or
// job -- it counts a skip and returns without running anything.
void* GapJobHandle(unsigned long long serial)
{
    return reinterpret_cast<void*>(static_cast<uintptr_t>((serial << 1) | 1));
}

struct Gap
{
    Microsoft::WRL::ComPtr<ID3D12Device> device;
    UINT64 nextValue = GapFenceBase;
    std::atomic<UINT64> signalled { 0 };
    bool ready = false;
    std::string failure;

    std::atomic<unsigned long long> runs { 0 };
    std::atomic<unsigned long long> submits { 0 };
    std::atomic<unsigned long long> failures { 0 };
    std::atomic<unsigned long long> vkSubmits { 0 };
    std::atomic<unsigned long long> vkFailures { 0 };
    std::atomic<unsigned long long> waits { 0 };
    // How many of the slot's submissions carried the frame's pixels, and how many carried the
    // handshake alone. The two not adding up to the number of runs is the interesting case.
    std::atomic<unsigned long long> copies { 0 };
    std::atomic<unsigned long long> noWork { 0 };
    // How many of those ran the network rather than the passthrough, and how many fell
    // back to it. The two not adding up to the number of copies is the interesting case.
    std::atomic<unsigned long long> netRuns { 0 };
    std::atomic<double> cpuTotalMs { 0.0 };
    std::atomic<double> cpuMaxMs { 0.0 };

    // The run split into the five steps it is made of: the game's queue told where the frame
    // has got to, the frame read back, the device waited for, the pass, and the answer sent
    // back. A run that is slow is slow in one of these, and which one it is cannot be told
    // from the total -- the total only says that one of them is.
    std::atomic<double> stepSignalMs { 0.0 };
    std::atomic<double> stepReadMs { 0.0 };
    // `wait` is how far behind the game's own queue was; `fence` is what the hand-over's
    // own copy cost. Both were one number before, and one number could not say which.
    std::atomic<double> stepWaitMs { 0.0 };
    std::atomic<double> stepFenceMs { 0.0 };
    std::atomic<double> stepWorkMs { 0.0 };
    std::atomic<double> stepWriteMs { 0.0 };

    /* The reference's Follow(), and the whole reason this block exists.
     *
     * MochizukiNrRuntime.cpp:7-8 declares the capability:
     *
     *   "Any game queue may run the list (MOCHIZUKI_NR_FEATURE_ANY_QUEUE): a frame from
     *    another queue than the last one waits for the last one first."
     *
     * and its Follow() (ta_MochizukiNrRuntime.cpp:2137) is what pays for it:
     *
     *   "A frame from another queue than the last frame's (by COM identity) first waits there
     *    for the last frame's finished value, so its signal cannot land before the last
     *    producer's (toVk would go backwards and meet a Vulkan wait early) and the Vulkan
     *    side takes the frames in order."
     *
     * We signal `value = nextValue++` on whichever queue the frame is on, and two D3D12
     * queues are independent: queue A's value 100 and queue B's value 101 can reach the GPU
     * in either order. If 101 lands first, Vulkan has already moved past 100 when 100
     * arrives -- toVk goes backwards -- and the network's answer belongs to a frame that is
     * not the one the slot copied. That is a wrong frame rather than a slow one, and it is
     * what a persistent flicker on a still scene is made of: standing still does not stop
     * the engine from handing successive frames to different queues.
     *
     * Nothing here is a theory of our own: the ordering requirement is the reference's, the
     * reason is the reference's, and the same hazard is the one this project already hit as
     * "cross-process semaphores do not order" and misattributed (see the project's memory:
     * the observation was right, the attribution was not). Two fences and a counter cannot
     * order two queues; only one of them waiting on the other can.
     *
     * ---------------------------------------------------------------------------
     * 2026-10-13 · 六块全部停用，但代码保留。
     *
     * 实测结论（本机、匹诺曹、2026-10-13 23:47-23:48 一次连续运行，日志有据）：
     *     NrBridge producer: queue=0x2d9454473b0 lastValue=505 | queue switches=0 of 505 runs (0.0%)
     * 505 次插槽运行，队列切换 **0** 次。《匹诺曹》全程单队列，ANY_QUEUE 这条路一次都不走。
     *
     * 停用理由不是它写错了 —— 逻辑逐条对着 ta_MochizukiNrRuntime.cpp:2137 的 Follow()
     * 写的，注释里的每一条理由都是引用。停用是因为**对这个游戏无效**，留着只会在每一帧
     * 白做一次 AddRef/Release（实测 runs 与 queue 完全相同即为空转）。
     *
     * 为什么留着不删：The 是靠这个能力位（MOCHIZUKI_NR_FEATURE_ANY_QUEUE，
     * ta_MochizukiNrRuntime.cpp:4512）支撑"任何队列都能跑"的，别的引擎（多队列提交
     * 更积极的、DX12 走的 UE4、多线程录制更狠的）**可能真的会换队列**。本游戏用不上
     * 不代表别的游戏用不上。等哪天在别的游戏上看到 `queueSwitches` 不为零，
     * 把 `#if 0` 去掉就能直接用，不必重写。
     *
     * 复用前必看两处曾经的坑：
     *   1. `ProducerId::Of` 必须 **AddRef**，不能 `ComPtr::Attach`。传进来的 queue 是
     *      BetweenThunk 借的裸指针，没人给它加过引用；Attach 会白拿一个不属于我们的引用，
     *      析构多 Release 一次，引用计数被多减 → 队列提前释放 → 实测崩在 cut #10。
     *   2. `resetPending` 必须在 `EnqueueGap` 里 Signal 之前判断，且消费点要在帧录制处
     *      （fused 走 freshHistory，非 fused 走 RestartHistory），不能省。
     * --------------------------------------------------------------------------- */
#if 0
    std::mutex producerMutex;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> lastProducer;
    IUnknown* lastProducerId = nullptr;
    // The last value this gap signalled and the value Vulkan finished, i.e. the two numbers
    // Follow() needs: what a new queue must wait for, and how far the old one got.
    std::atomic<UINT64> lastProduced { 0 };
    std::atomic<UINT64> lastFinished { 0 };
    // Set when a frame arrived on a different queue than the last one. The history is
    // restarted from it, because the reference says so in the same breath: "The new queue's
    // first half has already copied its colour, which may race the last frame's network:
    // both frames may look wrong, and the history restarts on the next frame."
    std::atomic<bool> resetPending { false };
    // The reference counts these and logs the first few, then one in a thousand, because
    // "a game may switch every frame" (its own comment). The count is also the measurement:
    // if it tracks the dropped-NR frames, this is the cause; if it stays near zero, it is not.
    std::atomic<unsigned long long> queueSwitches { 0 };
    std::atomic<unsigned long long> producerWaitFailures { 0 };
#endif
};

Gap& TheGap()
{
    // Process lifetime owner, the same way the reference keeps its Pending(): it is not torn
    // down while the game still has threads recording.
    static Gap g;
    return g;
}

// The reference's Producer::Of: a game queue and its COM identity, both kept, so that the
// comparison is against the object rather than against the address (a queue that has been
// destroyed and a new one made at the same address is a different queue).
//
// 🔒 2026-10-13 停用（实测 queue switches=0 of 505，本游戏单队列）。代码保留备用。
//    复用时注意下面 Of() 里 AddRef 那个坑。详见 Gap 结构体上方的说明块。
#if 0
struct ProducerId
{
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;
    IUnknown* id = nullptr;

    static ProducerId Of(ID3D12CommandQueue* q)
    {
        ProducerId p;
        if (q == nullptr)
            return p;
        /* AddRef, not Attach. The reference writes `q->AddRef()` here
         * (ta_MochizukiNrRuntime.cpp:1034) and the word is the whole of it: the pointer
         * handed to EnqueueGap is the game's, borrowed by BetweenThunk, and nobody has taken a
         * reference on it. Attach would take ownership of a reference that was never ours,
         * and the ComPtr's Release would then be one Release too many -- the queue's count
         * falls below the number of owners and the next thing that touches it is a dead
         * object. That is what ended the run at cut #10. */
        q->AddRef();
        p.queue.Attach(q);
        q->QueryInterface(IID_IUnknown, reinterpret_cast<void**>(&p.id));
        return p;
    }

    ~ProducerId()
    {
        if (id != nullptr)
            id->Release();
    }

    ProducerId() = default;
    // Moved, not copied: the identity is a reference, and two owners would release it twice.
    ProducerId(ProducerId&& other) noexcept
        : queue(std::move(other.queue)), id(std::exchange(other.id, nullptr))
    {
    }
    ProducerId& operator=(ProducerId&& other) noexcept
    {
        if (this != &other)
        {
            queue = std::move(other.queue);
            id = std::exchange(other.id, nullptr);
        }
        return *this;
    }
    ProducerId(const ProducerId&) = delete;
    ProducerId& operator=(const ProducerId&) = delete;
};
#endif

// The reference's Follow(), called once per slot run, before the producer's Signal.
//
// The order inside is the reference's, and every step of it has a reason stated there:
//
//   1. the same queue as last time: nothing to do;
//   2. a different queue: wait on it for the last frame's finished value, so this frame's
//      Signal cannot land before the last one's (toVk would go backwards);
//   3. a Signal nothing will finish (the frame failed after it) has no finished value to wait
//      for, so the new queue waits for it in toVk instead -- the reference handles this case
//      too, and it is the one that bites while frames are being dropped;
//   4. the queue is remembered for the next frame.
//
// Step 2 is the one whose absence let toVk go backwards. It is a queue waiting on a fence,
// which is the only thing that orders two independent D3D12 queues; two fences and a counter
// cannot.
//
// 🔒 2026-10-13 停用。逻辑本身是对的（逐条对着 ta_MochizukiNrRuntime.cpp:2137），
//    但实测本游戏 queue switches=0 of 505，一次都不触发，纯属每帧空转。
//    代码保留，换个会换队列的游戏去掉 #if 0 即可。详见 Gap 结构体上方说明块。
#if 0
HRESULT Follow(ID3D12CommandQueue* producer)
{
    Gap& g = TheGap();
    auto& vk = VulkanGap::TheBridge();
    std::lock_guard<std::mutex> lock(g.producerMutex);

    ProducerId p = ProducerId::Of(producer);
    if (p.id == nullptr)
        return E_INVALIDARG;

    // (1) The reference returns S_OK here, on the identity, having done nothing. Taken before
    // any of the bookkeeping below, so the common frame -- the same queue as the last, which
    // is every frame when the engine does not rotate -- costs one identity compare and one
    // pair of AddRef/Release, and touches no fence at all.
    if (g.lastProducerId != nullptr && p.id == g.lastProducerId)
        return S_OK;

    const UINT64 last = g.lastFinished.load(std::memory_order_relaxed);
    const UINT64 produced = g.lastProduced.load(std::memory_order_relaxed);
    // (3) `stray` is the reference's own name for a signalled value nothing finished: the
    // frame failed after its Signal and never reached Vulkan.
    const UINT64 signalled = g.signalled.load(std::memory_order_relaxed);
    const UINT64 stray = signalled > produced ? signalled : 0;

    if (last != 0 || stray != 0)
    {
        // (2) The wait that makes one queue's work ordered against the other's.
        HRESULT hr = last != 0 ? producer->Wait(vk.fromVk.Get(), last) : S_OK;
        if (SUCCEEDED(hr) && stray != 0)
            hr = producer->Wait(vk.toVk.Get(), stray);
        if (FAILED(hr))
        {
            g.producerWaitFailures.fetch_add(1, std::memory_order_relaxed);
            return hr;
        }

        // The reference, in the same breath as the wait: "both frames may look wrong, and
        // the history restarts on the next frame". Dropped where the frame is recorded.
        g.resetPending.store(true, std::memory_order_relaxed);

        const unsigned long long n = g.queueSwitches.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n <= 12 || n % 1000 == 0)
            LOG_INFO("NrBridge gap: a frame from queue {} after one from queue {} (switch {} of the gap): "
                     "it waits there for the last frame; this frame may look wrong, and the history restarts",
                     static_cast<void*>(producer), static_cast<void*>(g.lastProducer.Get()), n);
    }

    // (4) The reference replaces lastProducer under the same mutex.
    if (g.lastProducerId != nullptr)
        g.lastProducerId->Release();
    g.lastProducerId = p.id;
    p.id = nullptr;          // ownership moved to the gap
    g.lastProducer = p.queue;
    return S_OK;
}
#endif

// `on`, where it is not null, is the device the network runs on: the hand-over is
// then built on it rather than beside it, which is the shape the reference has (one
// Vulkan device and one queue for both halves of the bridge). Without it a second
// device of our own is made, and the two do not share a queue to be ordered on.
bool BuildGap(ID3D12Device* device, const VulkanGap::Borrowed* on = nullptr)
{
    Gap& g = TheGap();

    std::string why;
    const bool made = on != nullptr ? VulkanGap::CreateOn(device, *on, why)
                                    : VulkanGap::Create(device, why);
    if (!made)
    {
        g.failure = "VulkanGap: " + why;
        return false;
    }

    // The adapter is named, because "the game's own adapter" is a claim made from a LUID, and
    // the host spoofs adapters. If the vendor is not 0x8086 the import below was made against a
    // fence on another device, which is exactly the shape that waits forever.
    auto& up = VulkanGap::TheBridge();
    LOG_INFO("NrBridge: Vulkan hand-over up on {} (vendor 0x{:x} device 0x{:x}, queue family {}): the slot "
             "can now wait for the game and be waited on",
             up.deviceName, up.vendorId, up.deviceId, up.queueFamily);

    g.ready = true;
    return true;
}

// Where the network's pieces are: libxmx, the resident shaders, and the plan's own blob
// file, all of them written by the dump that recorded the plan. A directory rather than a
// search, because a network is a build and a build has to be the one that was checked.
std::filesystem::path NrFolder()
{
    const std::string configured = Config::Instance()->NrFolder.value_or_default();
    if (!configured.empty())
        return std::filesystem::path(configured);

    return std::filesystem::path("E:\\WorkBuddy\\2026-10-09-04-20-55\\src\\dlss-nr-on-a770\\work");
}

// Whether the pass is allowed to move the history by the frame's motion. The identity is
// what the layer has always had and is right wherever the scene stood still; this is the
// switch that says whether to try to be right where it moved too.
bool ReprojectHistory()
{
    return Config::Instance()->NrMvReproject.value_or_default();
}

// The reference's EnqueueHipFn, called by BetweenThunk between the two halves.
int32_t EnqueueGap(void* session, void* job, void* command_queue)
{
    if (job == nullptr || command_queue == nullptr)
        return -1;

    Gap& g = TheGap();
    const auto started = std::chrono::steady_clock::now();
    auto* queue = static_cast<ID3D12CommandQueue*>(command_queue);

    if (!g.ready)
    {
        if (FAILED(queue->GetDevice(IID_PPV_ARGS(&g.device))))
        {
            g.failure = "GetDevice";
            g.failures.fetch_add(1, std::memory_order_relaxed);
            return -1;
        }
        // The network's device, if there is one: the hand-over goes on it.
        auto* early = static_cast<NrBridge::Backend*>(session);
        NrBridge::DeviceHandles had {};
        VulkanGap::Borrowed on {};
        const bool borrowed = early != nullptr && early->NetworkDevice(&had);
        if (borrowed)
        {
            on.instance = reinterpret_cast<VkInstance>(had.instance);
            on.physical = reinterpret_cast<VkPhysicalDevice>(had.physical);
            on.device = reinterpret_cast<VkDevice>(had.device);
            on.queue = reinterpret_cast<VkQueue>(had.queue);
            on.family = had.family;
        }
        // Said once, because the two shapes look alike and only one of them reads the
        // frame: two devices of our own, each importing the same shared resource, do not
        // share a queue to be ordered on, and the answer comes back wrong.
        LOG_INFO("NrBridge: the hand-over stands on {}",
                 borrowed ? "the network's device" : "a device of its own");

        if (!BuildGap(g.device.Get(), borrowed ? &on : nullptr))
        {
            g.failures.fetch_add(1, std::memory_order_relaxed);
            return -1;
        }
    }

    const unsigned long long run = g.runs.fetch_add(1, std::memory_order_relaxed) + 1;
    const bool announce = run <= 20;

    // The reference's own order, and the reason for it: the game's queue says where the frame
    // has got to (Signal on the fence Vulkan waits on), Vulkan runs, and Vulkan's own signal
    // is what the game's queue is later made to wait on. The hand-over fence is the one the
    // bridge imported, so the value enqueued here is the value the Vulkan queue waits for --
    // the frame's producer half completes before any of this runs.
    // The reference's Follow(), before anything is signalled: a frame from a queue other than
    // the last frame's waits there for the last frame, so its Signal cannot land first. See
    // the block comment on Gap's producer fields for why this is not optional.
#if 0
    // 🔒 2026-10-13 停用：实测本游戏 queue switches=0 of 505（单队列），本调用恒为空转。
    //    需要时连同 Follow()、Gap 的 producer 字段一起取消注释，三处缺一不可。
    if (FAILED(Follow(queue)))
    {
        g.producerWaitFailures.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }
#endif

    const UINT64 value = g.nextValue++;
    auto& vk = VulkanGap::TheBridge();
    const auto tSignal = Clock::now();
    const HRESULT hr = queue->Signal(vk.toVk.Get(), value);
    const auto tRead = Clock::now();
    g.signalled.store(value, std::memory_order_relaxed);
#if 0
    // The reference records the produced value under the same mutex as Follow's, which is
    // what makes the two agree on which value is the last one. Ours is a single writer on
    // the render thread, which is the same guarantee said shorter.
    // 🔒 2026-10-13 停用，随 Follow() 一起（Gap 的 lastProduced 已注释）。
    g.lastProduced.store(value, std::memory_order_relaxed);
#endif

    if (FAILED(hr))
    {
        g.failure = "Signal(toVk)";
        g.failures.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }

    // What the slot runs: the network where the reference has its renderer, or the
    // reference's Passthrough where there is none.
    //
    // The network needs two submissions, because the host's work is between them and cannot
    // be inside either: one that carries the frame's pixels to a buffer the host can read,
    // and one that carries the answer back. The first signals nothing -- it is not the
    // answer, it is the question -- and between the two the host waits for the device once
    // and does the pass.
    //
    // A pass that will not come up, or whose submission will not go, falls back to the
    // passthrough rather than to nothing: a frame the slot does not answer is a frame whose
    // continuation half reads a buffer nobody wrote.
    auto* host = static_cast<NrBridge::Backend*>(session);
    std::string why;

    bool answered = false;
    VulkanGap::Transfer transfers[2] {};
    UINT moving = 0;

    if (host != nullptr && Config::Instance()->NrNetwork.value_or_default() && host->SlotRead(transfers, moving))
    {
        const VkResult read = VulkanGap::Submit(value, transfers, moving, false, why);
        const auto tWait = Clock::now();

        // Two waits, timed apart. The first is the game's queue getting to the frame's
        // producer half; the second is the device finishing the copy the slot just handed
        // it. Both are inside the one `wait` the log used to print, and they are not the
        // same wait: one is how far behind the game is, the other is what the hand-over
        // itself costs.
        const bool reached = read == VK_SUCCESS && VulkanGap::WaitGame(value, why) == VK_SUCCESS;
        const auto tFence = Clock::now();
        const bool waited = reached && VulkanGap::WaitLast(why) == VK_SUCCESS;
        const auto tWork = Clock::now();

        if (waited && host->SlotWork(value))
        {
            const auto tWrite = Clock::now();
            UINT writing = 0;
            if (host->SlotWrite(transfers, writing))
            {
                if (writing != 0)
                {
                    moving = writing;
                    const VkResult wrote = VulkanGap::Submit(value, transfers, moving, true, why);
                    answered = wrote == VK_SUCCESS;
                    if (!answered)
                    {
                        g.vkFailures.fetch_add(1, std::memory_order_relaxed);
                        g.failure = why + " (" + std::to_string(static_cast<int>(wrote)) + ")";
                    }
                }
                else
                {
                    // The fused path: the network wrote the answer into the game's own
                    // buffer, so there is nothing to carry back -- but the handshake is
                    // still owed. Mochi's shape is that the network runs on the Vulkan
                    // queue after the value the game signalled and signals the next
                    // (docs/ARCHITECTURE.md), which is this submit with no copies in it.
                    //
                    // Without the submit nothing signals `fromVk` and the game's queue
                    // waits for a value that will never arrive; with the passthrough
                    // instead, the input is copied over what the network just wrote.
                    const VkResult wrote = VulkanGap::Submit(value, nullptr, 0, true, why);
                    answered = wrote == VK_SUCCESS;
                    if (!answered)
                    {
                        g.vkFailures.fetch_add(1, std::memory_order_relaxed);
                        g.failure = why + " (" + std::to_string(static_cast<int>(wrote)) + ")";
                    }
                }
            }
            const auto tDone = Clock::now();
            g.stepSignalMs.store(ElapsedMs(tSignal, tRead), std::memory_order_relaxed);
            g.stepReadMs.store(ElapsedMs(tRead, tWait), std::memory_order_relaxed);
            g.stepWaitMs.store(ElapsedMs(tWait, tFence), std::memory_order_relaxed);
            g.stepFenceMs.store(ElapsedMs(tFence, tWork), std::memory_order_relaxed);
            g.stepWorkMs.store(ElapsedMs(tWork, tWrite), std::memory_order_relaxed);
            g.stepWriteMs.store(ElapsedMs(tWrite, tDone), std::memory_order_relaxed);
        }
    }

    if (answered)
    {
        g.copies.fetch_add(1, std::memory_order_relaxed);
        g.netRuns.fetch_add(1, std::memory_order_relaxed);
    }
    else
    {
        // The reference's Passthrough: one copy, in the network's place. It is asked for
        // before the submit because the answer decides whether the submit carries work at
        // all -- no buffers, no copy, and the handshake on its own is still what gets
        // measured.
        VkBufferCopy copy {};
        VkBuffer from = VK_NULL_HANDLE, to = VK_NULL_HANDLE;
        const bool work = host != nullptr && host->SlotCopy(&copy, &from, &to);
        if (work)
            g.copies.fetch_add(1, std::memory_order_relaxed);
        else
            g.noWork.fetch_add(1, std::memory_order_relaxed);

        transfers[0].from = from;
        transfers[0].to = to;
        transfers[0].region = copy;

        const VkResult vr = VulkanGap::Submit(value, work ? transfers : nullptr, work ? 1u : 0u, true, why);
        if (vr != VK_SUCCESS)
        {
            // Recorded, and left for the next frame: a Wait on a value nothing signalled
            // would stop the game's queue rather than slow it, so nothing is queued for it
            // here.
            //
            // The reason is kept, not accumulated. A reason appended on every failure makes
            // the log line grow with the frame count, and a log line that grows is a log
            // that stops being readable exactly when the failure it describes is being
            // looked for.
            g.vkFailures.fetch_add(1, std::memory_order_relaxed);
            g.failure = why + " (" + std::to_string(static_cast<int>(vr)) + ")";
            g.failures.fetch_add(1, std::memory_order_relaxed);
            return -1;
        }
    }

    g.vkSubmits.fetch_add(1, std::memory_order_relaxed);
    g.submits.fetch_add(1, std::memory_order_relaxed);

    // The reference closes the ring in the same callback, straight after the submit goes through:
    // the game's queue waits for the value the Vulkan queue is going to signal. It asks for that
    // wait only once `submitted` is true, because a Wait on a value nothing will signal stops the
    // queue instead of slowing it -- which is why the failed-return above queues nothing.
    const HRESULT waited = queue->Wait(vk.fromVk.Get(), value);
    if (FAILED(waited))
    {
        char text[64];
        std::snprintf(text, sizeof text, "Wait(fromVk) 0x%08lX", static_cast<unsigned long>(waited));
        g.failure = text;
        g.failures.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }

    g.waits.fetch_add(1, std::memory_order_relaxed);

    // The reference's Submitted(): the frame is on the Vulkan queue, so fromVk will reach
    // this value whether or not the wait above went through, and a later frame on another
    // queue is owed a wait for exactly this number. Recorded only here -- after the submit
    // and the wait are both queued -- because a value recorded before them is a value
    // nothing has promised yet, and Follow() would wait on a promise that never came.
#if 0
    // 🔒 2026-10-13 停用，随 Follow() 一起（Gap 的 lastFinished 已注释）。
    g.lastFinished.store(value, std::memory_order_release);
#endif

    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    g.cpuTotalMs.store(g.cpuTotalMs.load(std::memory_order_relaxed) + ms, std::memory_order_relaxed);
    double peak = g.cpuMaxMs.load(std::memory_order_relaxed);
    while (ms > peak && !g.cpuMaxMs.compare_exchange_weak(peak, ms, std::memory_order_relaxed))
    {
    }

    // Announced for the first few runs and then every fiftieth: the first run carries the
    // network's own build and says nothing about a frame, and a run every fifty is close
    // enough to watch a number that is drifting without filling the log with it.
    if (announce || run % 50 == 0)
    {
        // The pass's own split travels in the same line: the device's wait and the host's
        // work are the two halves of `work` above, and seeing them beside it is what tells
        // a slow pass from a pass that waited.
        const double hostMs = host != nullptr
                                  ? host->LastCopyMs() + host->LastDecodeMs() + host->LastMotionMs() +
                                        host->LastNetMs() + host->LastFeaturesMs() + host->LastComposeMs() +
                                        host->LastEncodeMs()
                                  : 0.0;
        LOG_INFO("NrBridge gap run #{}: queue={} value={} took={:.3f}ms | signal={:.3f} read={:.3f} wait={:.3f} "
                 "fence={:.3f} work={:.3f} write={:.3f} | copy={:.3f} decode={:.3f} mv={:.3f} features={:.3f} "
                 "net={:.3f} compose={:.3f} encode={:.3f} (host {:.3f})",
                 run, command_queue, value, ms,
                 g.stepSignalMs.load(std::memory_order_relaxed), g.stepReadMs.load(std::memory_order_relaxed),
                 g.stepWaitMs.load(std::memory_order_relaxed), g.stepFenceMs.load(std::memory_order_relaxed),
                 g.stepWorkMs.load(std::memory_order_relaxed), g.stepWriteMs.load(std::memory_order_relaxed),
                 host != nullptr ? host->LastCopyMs() : 0.0, host != nullptr ? host->LastDecodeMs() : 0.0,
                 host != nullptr ? host->LastMotionMs() : 0.0, host != nullptr ? host->LastFeaturesMs() : 0.0,
                 host != nullptr ? host->LastNetMs() : 0.0, host != nullptr ? host->LastComposeMs() : 0.0,
                 host != nullptr ? host->LastEncodeMs() : 0.0, hostMs);

        // The graph's own split, read at the same cadence because the read clears the
        // span: fifty frames of submit/wait/collect averaged into one line is the number
        // the decision needs, and reading anywhere else would steal frames from the line
        // that prints them. The three are the parts of the one `work`/`net` block above:
        // the CPU handing the graph over, the CPU blocked in vkWaitForFences, and the
        // timestamp read-back -- plus the staging transfers in front of it (the frame's
        // exposure), which the submit's three do not include.
        double t[12] {};
        if (host != nullptr && host->SubmitTimings(t) && t[0] > 0.0)
        {
            const double frames = t[0];
            LOG_INFO("NrBridge gap graph #{}: over {:.0f} submits | submit={:.3f} wait={:.3f} "
                     "collect={:.3f} mean={:.3f} ms/frame | max {:.3f}/{:.3f}/{:.3f} | "
                     "stage={:.3f}ms total, slowest {:.3f}, {:.0f} calls",
                     run, frames, t[1] / frames, t[3] / frames, t[5] / frames, t[7],
                     t[2], t[4], t[6], t[8], t[9], t[10]);
        }
    }

    return 0;
}

void ReportSplitEligibility(ID3D12GraphicsCommandList* cmdList, unsigned long long n)
{
    const bool early = n < 6;
    const bool periodic = (n % 200) == 0;
    if (!early && !periodic)
        return;

    DlssNr::Submission::ILogicalCommandList* logical = nullptr;
    const auto qi = cmdList->QueryInterface(__uuidof(DlssNr::Submission::ILogicalCommandList),
                                            reinterpret_cast<void**>(&logical));
    if (logical == nullptr)
    {
        LOG_INFO("NrBridge frame {}: list {} is not one of ours (qi 0x{:x})", n,
                 static_cast<void*>(cmdList), static_cast<unsigned>(qi));
        return;
    }

    const bool ineligible = logical->IsSplitIneligible();
    const char* why = logical->SplitRejectionReason();
    LOG_INFO("NrBridge frame {}: list {} proxied, splitIneligible={} reason={}", n,
             static_cast<void*>(cmdList), ineligible, why != nullptr ? why : "(none)");
    logical->Release();
}

NrBridge::Frame ReadFrame(NVSDK_NGX_Parameter* params)
{
    NrBridge::Frame f {};

    f.colour = ReadResource(params, NVSDK_NGX_Parameter_Color);
    f.motion = ReadResource(params, NVSDK_NGX_Parameter_MotionVectors);
    f.depth = ReadResource(params, NVSDK_NGX_Parameter_Depth);
    f.exposure = ReadResource(params, NVSDK_NGX_Parameter_ExposureTexture);

    f.width = ReadUInt(params, NVSDK_NGX_Parameter_Width);
    f.height = ReadUInt(params, NVSDK_NGX_Parameter_Height);
    f.motionScaleX = ReadFloat(params, NVSDK_NGX_Parameter_MV_Scale_X, 1.0f);
    f.motionScaleY = ReadFloat(params, NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);
    f.jitterX = ReadFloat(params, NVSDK_NGX_Parameter_Jitter_Offset_X, 0.0f);
    f.jitterY = ReadFloat(params, NVSDK_NGX_Parameter_Jitter_Offset_Y, 0.0f);
    f.preExposure = ReadFloat(params, NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1.0f);
    f.reset = ReadUInt(params, NVSDK_NGX_Parameter_Reset) != 0;

    // The vectors here are the same size as the colour in this game, and the reference records
    // the extent separately because that is not true everywhere. Left at zero, which is what
    // "render-resolution vectors" means there.
    f.motionWidth = 0;
    f.motionHeight = 0;

    // Whether the vectors already carry the jitter, and which way the depth runs. The host knows
    // both from the flags the upscaler was created with, so they are read from it rather than
    // guessed: a temporal pass that unjitters twice is worse than one that does not.
    auto* feature = State::Instance().currentFeature;
    f.motionJittered = feature != nullptr && feature->JitteredMV();
    f.depthInverted = feature != nullptr && feature->DepthInverted();
    f.temporalInputsValid = f.motion != nullptr && f.depth != nullptr;

    f.colourState = ResolveState(Config::Instance()->ColorResourceBarrier, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    f.motionState = ResolveState(Config::Instance()->MVResourceBarrier, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    f.depthState = ResolveState(Config::Instance()->DepthResourceBarrier,
                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    f.exposureState = ResolveState(Config::Instance()->ExposureResourceBarrier,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    return f;
}

void LogFrame(const NrBridge::Frame& f)
{
    const auto n = g_seen.load(std::memory_order_relaxed);
    const bool early = n < 6;
    const bool periodic = (n % 200) == 0;

    if (!early && !periodic)
        return;

    LOG_INFO("NrBridge frame {} {}x{} mv=({},{}) jitter=({:.3f},{:.3f}) preExp={:.3f} reset={} | {} | {} | {}",
             n, f.width, f.height, f.motionScaleX, f.motionScaleY, f.jitterX, f.jitterY, f.preExposure,
             f.reset ? 1 : 0, Describe("color", f.colour), Describe("mv", f.motion),
             Describe("depth", f.depth));

    if (!g_contractLogged.exchange(true, std::memory_order_relaxed))
    {
        // Printed once, because it is the sentence this whole host exists for. If this says the
        // vectors are absent, nothing downstream is allowed to proceed.
        LOG_INFO("NrBridge contract: motion={} depth={} exposure={} backend={}", f.motion != nullptr,
                 f.depth != nullptr, f.exposure != nullptr,
                 Config::Instance()->Dx12Upscaler.value_or_default());

        if (f.motion == nullptr)
            LOG_WARN("NrBridge: no motion vectors in this evaluate -- temporal work must fall back");
    }

    // Printed with the reference's own field names, from its LmxxfBackend::RecordDiagnostic
    // line, because the question "did both halves of the cut actually reach the queue" is the
    // question that line exists to answer:
    //
    //   lmxxf boundary: eval={} proxy={} cutHr={:X} proxyHits={} cutsRecorded={} rejected={}
    //   unsplitSubmitted={} producerSubmitted={} continuationSubmitted={} submitFailures={}
    //   (counts are NOT GPU completion)
    //
    // producerSubmitted and continuationSubmitted are the two that matter here, and they are
    // the reference's own counts of its two Execute calls, not anything of ours. The between
    // slot's own counters are deliberately not used to judge this: with nothing enqueued the
    // reference's BetweenThunk returns before it counts anything, so they read zero whether or
    // not the slot ran.
    // `declined` is on this line because a frame the game would not let us cut is a frame
    // that shows the game its own colour instead of the network's answer, and that is the
    // whole difference between a picture that is denoised and one that flickers. The count
    // was being kept and never printed, which is why the flicker had no number on it.
    LOG_INFO("NrBridge boundary: eval={} cutHr={:X} proxyHits={} cutsRecorded={} rejected={} declined={} "
             "unsplitSubmitted={} producerSubmitted={} continuationSubmitted={} submitFailures={} "
             "(counts are NOT GPU completion)",
             n, static_cast<unsigned>(g_lastCutHr.load(std::memory_order_relaxed)),
             g_splitOk.load(std::memory_order_relaxed) + g_splitFailed.load(std::memory_order_relaxed),
             g_splitOk.load(std::memory_order_relaxed), g_splitFailed.load(std::memory_order_relaxed),
             g_splitNoProxy.load(std::memory_order_relaxed),
             DlssNr::Submission::g_unsplitProxySubmissions.load(std::memory_order_relaxed),
             DlssNr::Submission::g_splitSubmissions.load(std::memory_order_relaxed),
             DlssNr::Submission::g_continuationSubmissions.load(std::memory_order_relaxed),
             DlssNr::Submission::g_submissionFailures.load(std::memory_order_relaxed));

    /* The same decline, counted four ways. `declined` above is their sum, and which of them it is
     * decides what is worth doing: `notOurs` is the game's own list and only an earlier hook
     * reaches it; the other three are the proxy refusing a list it *did* wrap, which no amount of
     * hooking can fix and which needs the cut to become possible instead. Four numbers, one line,
     * so that the next run names the reason instead of leaving it to be guessed at. */
    LOG_INFO("NrBridge declines: notOurs={} ineligible={} noResources={} noCompute={} | "
             "notOurs is the game's own list (hook timing); the other three are the proxy "
             "refusing its own list (the cut has to become possible)",
             g_declNotOurs.load(std::memory_order_relaxed),
             g_declIneligible.load(std::memory_order_relaxed),
             g_declNoResources.load(std::memory_order_relaxed),
             g_declNoCompute.load(std::memory_order_relaxed));

    // Whether the slot runs at all. Its own counts come from the reference's Pending(), which
    // BetweenThunk fills in: betweenHits for a run whose list matched, enqueueCalls for the
    // call into the function, skippedHits for a run that found nothing to do.
    auto& pending = DlssNr::Backend::LmxxfCut::Pending();
    Gap& gap = TheGap();

    // Both halves of the hand-over, read back from the fences themselves. toVk is the fence the
    // game's queue signals and Vulkan waits on: its completed value catching the last value
    // queued is the GPU reaching the gap in order. fromVk is the one Vulkan signals: its value
    // climbing is the Vulkan queue having run and handed the frame back, and it is the value the
    // game's queue is now made to wait on in the slot, which closes the ring.
    auto& vk = VulkanGap::TheBridge();
    const UINT64 toVkDone = vk.toVk != nullptr ? vk.toVk->GetCompletedValue() : 0;
    const UINT64 fromVkDone = vk.fromVk != nullptr ? vk.fromVk->GetCompletedValue() : 0;
    const UINT64 signalled = gap.signalled.load(std::memory_order_relaxed);

    // toVkDone and gapSignalled are the same fence read by its two owners; seenToVk is what
    // Vulkan makes of it. If seenToVk stays behind toVkDone, the import never shared the fence,
    // and the Vulkan side will wait for a value nothing is going to give it.
    LOG_INFO("NrBridge gap: between={} enqueue={} skipped={} | runs={} submits={} waits={} failures={} cpu={:.3f}ms "
             "max={:.3f}ms{} | vk submits={} failures={} copies={} nowork={}{} | toVkDone={} seenByVk={} | fromVkDone={} "
             "seenByVk={} gapSignalled={}",
             pending.betweenHits.load(std::memory_order_relaxed),
             pending.enqueueCalls.load(std::memory_order_relaxed),
             pending.skippedHits.load(std::memory_order_relaxed),
             gap.runs.load(std::memory_order_relaxed), gap.submits.load(std::memory_order_relaxed),
             gap.waits.load(std::memory_order_relaxed),
             gap.failures.load(std::memory_order_relaxed), gap.cpuTotalMs.load(std::memory_order_relaxed),
             gap.cpuMaxMs.load(std::memory_order_relaxed), gap.failure.empty() ? "" : gap.failure.c_str(),
             gap.vkSubmits.load(std::memory_order_relaxed), gap.vkFailures.load(std::memory_order_relaxed),
             gap.copies.load(std::memory_order_relaxed), gap.noWork.load(std::memory_order_relaxed),
             vk.lost ? " (vk given up on after " + std::to_string(vk.timeouts) + " timeouts)" : "",
             toVkDone, vk.seenToVk, fromVkDone, vk.seenFromVk, signalled);

    #if 0
    /* The producer's queue, and how often it changes. This is the measurement the whole
     * Follow() mechanism exists for, so it is read as a rate against the same window as the
     * rest of the line rather than as a bare total: `switches` against `runs` says whether
     * this game hands successive frames to different queues often enough to matter, and a
     * number that stays at zero says the mechanism is armed and idle, which is a different
     * answer from one that climbs.
     *
     * 🔒 2026-10-13 停用：测量已做完，结论是「本游戏单队列，switches=0 of 505」，
     *    所以 Follow() 一并停用（见 Gap 结构体上方说明块）。留着这行是为了
     *    将来在别的游戏上恢复该机制时，判据不用重写。 */
    const auto runs = gap.runs.load(std::memory_order_relaxed);
    const auto switches = gap.queueSwitches.load(std::memory_order_relaxed);
    LOG_INFO("NrBridge producer: queue={} lastValue={} | queue switches={} of {} runs ({:.1f}%) waitFailures={}",
             static_cast<void*>(gap.lastProducer.Get()), gap.lastProduced.load(std::memory_order_relaxed), switches,
             runs, runs != 0 ? (100.0 * double(switches) / double(runs)) : 0.0,
             gap.producerWaitFailures.load(std::memory_order_relaxed));
#endif

    // What the pass itself did, on the frame it last ran. Printed beside the slot's own
    // counts because the two answer different questions: the slot's say whether the gap
    // ran, and these say what the network answered when it did. A gate near 0.7 is a
    // history the model is accepting; near 0.03 is one it is rejecting, which on a scene
    // that moves is what a wrong reprojection looks like.
    auto* backend = g_backend.load(std::memory_order_acquire);
    if (backend != nullptr)
    {
        const char* refused = backend->NetworkFailure();
        const std::string state = backend->NetworkUp() ? "up"
                                  : (refused != nullptr && *refused != '\0') ? refused
                                                                             : "never tried";
        LOG_INFO("NrBridge pass: network={} copy={:.2f} decode={:.2f} mv={:.2f} features={:.2f} net={:.2f} "
                 "compose={:.2f} encode={:.2f}ms | gate={:.3f} head mean={:.4f} peak={:.1f} | colour in [{:.3f}, "
                 "{:.3f}]",
                 state, backend->LastCopyMs(), backend->LastDecodeMs(), backend->LastMotionMs(),
                 backend->LastFeaturesMs(), backend->LastNetMs(), backend->LastComposeMs(), backend->LastEncodeMs(),
                 backend->LastGate(),
                 backend->LastHeadMean(), backend->LastHeadPeak(), backend->LastLow(), backend->LastHigh());
    }
}

// Built on first use and never rebuilt, matching the reference: it builds what it needs on first
// use and disables itself for the session if anything fails, rather than retrying into a crash.
NrBridge::Backend* EnsureBackend(ID3D12CommandQueue* timingQueue)
{
    auto* existing = g_backend.load(std::memory_order_acquire);
    if (existing != nullptr)
        return existing;

    std::lock_guard<std::mutex> guard(g_backendMutex);

    existing = g_backend.load(std::memory_order_relaxed);
    if (existing != nullptr)
        return existing;

    auto* queue = timingQueue != nullptr ? timingQueue : State::Instance().currentCommandQueue;
    if (queue == nullptr)
    {
        // Not a failure: the queue only exists once a swapchain does. Try again on a later frame.
        return nullptr;
    }

    ID3D12Device* device = nullptr;
    if (FAILED(queue->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void**>(&device))) ||
        device == nullptr)
    {
        g_failureReason = "no device from the command queue";
        g_failed.store(true, std::memory_order_release);
        LOG_WARN("NrBridge: {} -- pass disabled for this session", g_failureReason);
        return nullptr;
    }

    // Before anything else is built: the lists the game makes from now on can be cut, the one
    // it is recording now cannot.
    EnsureSubmissionArmed(device, queue);

    // The hand-over first, before the backend records a frame. The frame's buffers are named on
    // the Vulkan device, so they can only be made once there is one, and this device already
    // exists here while the slot -- which is where this used to be built, on its first run --
    // has not run yet. A failure is not fatal and not final: the slot builds it again if it
    // gets there, and until then frames go through untouched.
    std::string why;
    if (!VulkanGap::Create(device, why))
        LOG_WARN("NrBridge: the Vulkan hand-over is not up yet ({}): frames go through until the slot can build it",
                 why);

    auto* backend = new NrBridge::Backend(device, queue, Util::DllPath().parent_path());
    g_backend.store(backend, std::memory_order_release);
    LOG_INFO("NrBridge: backend created on queue {}", static_cast<void*>(queue));
    return backend;
}
} // namespace

namespace NrBridge
{
struct Backend::Impl
{
    ID3D12Device* device = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    std::filesystem::path directory;
    std::atomic<unsigned long long> recorded { 0 };

    // Everything sized by the frame: the two shared buffers the colour crosses in, and the
    // texture the upscaler is handed back. Made again when the frame's extent or format
    // changes, which with dynamic resolution it does, and never sized from anything but the
    // frame that is being recorded.
    UINT width = 0, height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    DXGI_FORMAT motionFormat = DXGI_FORMAT_UNKNOWN;
    UINT mvWidth = 0, mvHeight = 0;
    NrBridge::Footprint footprint;

    // The motion vectors cross in a buffer of their own: their format is not the colour's, so
    // their footprint is not either, and a copy described with the colour's would tear them.
    // Their extent is theirs too, and is taken from the same place the colour's is taken from
    // when the frame does not report one: the texture itself.
    NrBridge::Footprint mvFootprint;
    NrBridge::SharedBuffer input, output;
    NrBridge::SharedBuffer mvInput;
    Microsoft::WRL::ComPtr<ID3D12Resource> result;

    // The state the result texture is actually in, as far as this module's own transitions go:
    // a texture does not decay the way a buffer does, so what it was left in by the last frame
    // is what the next barrier must name as its StateBefore.
    D3D12_RESOURCE_STATES resultState = D3D12_RESOURCE_STATE_COMMON;

    // The set that was in use before the last change of geometry, kept because the GPU may
    // still be reading it: what frees it is the next change, which is at least one frame
    // later, and a frame's work has been waited on by then.
    NrBridge::SharedBuffer oldInput, oldOutput, oldMvInput;
    Microsoft::WRL::ComPtr<ID3D12Resource> oldResult;

    // The network, and the buffers the host works in.
    //
    // The shared buffers the frame crosses in are device-local, which is what importing a
    // D3D12 allocation asks for and what the host cannot read. So a frame that the network
    // is to see crosses twice more, into and out of three of its own: colour and motion on
    // the way in, the composed frame on the way out.
    NrBridge::Pipeline network;
    bool networkTried = false;
    std::string networkFailure;

    VulkanGap::Staging stageColour, stageMotion, stageResult;
    std::vector<float> hostColour, hostMotion, hostResult;

    // The same three frames again, in the host's own memory, and the reason they exist:
    // a staging buffer is the device's memory mapped through BAR, and reading it one pixel
    // at a time is a PCIe round trip per pixel -- measured at 230 ns a pixel, which is
    // three quarters of the frame's time, against the 1.6 ms the same number of pixels
    // costs in memory the host owns. So the frame is taken across in one copy, worked on
    // here, and taken back in one copy: the passes never touch a mapped pointer.
    std::vector<uint8_t> rawColour, rawMotion, rawResult;

    // The frame's motion, as the host needs it: the vectors are uv, and the shift they
    // stand for is in pixels of the frame's own extent -- the reference's own reading of
    // them, `motionScaleX = motion_scale_x / motionWidth`, times that extent.
    float pixelsPerUvX = 0, pixelsPerUvY = 0;
    float preExposure = 1;
    bool freshHistory = true;
    bool haveMotion = false;

    // What the pass cost, and what it answered, for the log.
    double lastFeaturesMs = 0, lastNetMs = 0, lastComposeMs = 0;
    // What the pass spends around the network: the frame out of the game's format, the vectors
    // out of theirs, and the answer back into the game's. Timed separately from the network
    // because they are the host's own work, and because a frame that is slow for one of these
    // reasons is slow in a different place from one that is slow for the network's.
    double lastDecodeMs = 0, lastMotionMs = 0, lastEncodeMs = 0;
    // What the two one-copy crossings cost. Kept apart from the three above: if this is
    // where the time went instead, the frame did not get cheaper, it only moved.
    double lastCopyMs = 0;
    float lastGate = 0, lastHeadMean = 0, lastHeadPeak = 0;
    float lastLow = 0, lastHigh = 0;

    std::string bufferFailure;
    bool BuildBuffers(ID3D12Resource* colour, ID3D12Resource* motion, UINT w, UINT h);
    bool EnsureNetwork(UINT w, UINT h);
};

// The frame's buffers, made the first time and whenever the frame's geometry is not the one
// they were made for. A failure is recorded and the frame is left alone, which is the same
// answer the reference gives: pass the frame through rather than fail it.
bool Backend::Impl::BuildBuffers(ID3D12Resource* colour, ID3D12Resource* motion, UINT w, UINT h)
{
    if (colour == nullptr || w == 0 || h == 0)
    {
        bufferFailure = "the frame has no extent";
        return false;
    }

    const auto desc = colour->GetDesc();

    // The motion vectors' format is part of what the buffers are made for: a frame whose motion
    // changed format without changing extent has to be made again all the same.
    const DXGI_FORMAT mvFormat = motion != nullptr ? motion->GetDesc().Format : DXGI_FORMAT_UNKNOWN;
    const UINT mvW = motion != nullptr ? static_cast<UINT>(motion->GetDesc().Width) : 0;
    const UINT mvH = motion != nullptr ? static_cast<UINT>(motion->GetDesc().Height) : 0;
    // `input.resource`, not `input.buffer`, and the difference is the whole frame.
    //
    // `buffer` is the handle's image on the hand-over's own Vulkan device, and the fused
    // path deliberately never makes one: the network's device imports the shared handle
    // instead (VulkanGap::CreateOn). So this test could never be true under fused, and
    // every frame fell through and rebuilt all four buffers. A rebuild makes a new D3D12
    // resource and a new shared handle, while `nr_net_fuse` -- asked once, before the
    // open -- still holds the handle the first frame had. The network therefore reads a
    // resource nobody writes any more, and the colour it reads never changes: that is
    // what a fixed `in mean` of 0.1731 in the log was, across a moving camera.
    if (input.resource != nullptr && width == w && height == h && format == desc.Format &&
        motionFormat == mvFormat && mvWidth == mvW && mvHeight == mvH)
        return true;

    oldInput.Release();
    oldOutput.Release();
    oldMvInput.Release();
    oldResult.Reset();

    oldInput = input;
    oldOutput = output;
    oldMvInput = mvInput;
    oldResult = result;

    input = {};
    output = {};
    mvInput = {};
    result.Reset();
    resultState = D3D12_RESOURCE_STATE_COMMON;

    NrBridge::Footprint fp(desc.Format, w, h);
    if (!fp.valid())
    {
        bufferFailure = "the colour format has no footprint";
        return false;
    }

    std::string why;
    if (!input.Create(device, fp.bytes, why))
    {
        bufferFailure = "input " + why;
        return false;
    }
    if (!output.Create(device, fp.bytes, why))
    {
        bufferFailure = "output " + why;
        return false;
    }

    // The motion vectors get a buffer of their own, described by a footprint of their own: the
    // copy that fills it is a texture copy, and one described with the colour's footprint would
    // read them at the colour's texel size. A frame that has no motion, or whose format has no
    // footprint, keeps its colour path -- the temporal work is what goes without, and deciding
    // what to do about that is not this module's.
    if (motion != nullptr)
    {
        NrBridge::Footprint mfp(mvFormat, mvW, mvH);
        if (!mfp.valid())
        {
            static std::atomic<bool> saidMvFormat { false };
            if (!saidMvFormat.exchange(true, std::memory_order_relaxed))
                LOG_WARN("NrBridge: the motion format {} has no footprint; motion is not carried",
                         static_cast<int>(mvFormat));
        }
        else if (!mvInput.Create(device, mfp.bytes, why))
        {
            static std::atomic<bool> saidMvBuffer { false };
            if (!saidMvBuffer.exchange(true, std::memory_order_relaxed))
                LOG_WARN("NrBridge: the motion buffer could not be made ({}); motion is not carried", why);
            mfp = NrBridge::Footprint();
        }
        mvFootprint = mfp;
    }
    else
    {
        mvFootprint = NrBridge::Footprint();
    }

    D3D12_HEAP_PROPERTIES hp {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;

    const D3D12_RESOURCE_DESC rd = ResultDesc(desc, w, h);
    if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&result))) ||
        !result)
    {
        bufferFailure = "could not create the result texture";
        result.Reset();
        return false;
    }

    // The host's own three, sized by the same footprints. A frame whose motion is not
    // carried works without one: the pass runs, and its history stands still.
    std::string stagingFailure;
    if (!stageColour.Create(fp.bytes, stagingFailure))
    {
        bufferFailure = "staging " + stagingFailure;
        return false;
    }
    if (!stageResult.Create(fp.bytes, stagingFailure))
    {
        bufferFailure = "staging " + stagingFailure;
        return false;
    }
    stageMotion.Release();
    hostMotion.clear();
    if (mvFootprint.valid() && mvInput.resource != nullptr && !stageMotion.Create(mvFootprint.bytes, stagingFailure))
    {
        static std::atomic<bool> saidMotionStage { false };
        if (!saidMotionStage.exchange(true, std::memory_order_relaxed))
            LOG_WARN("NrBridge: the motion staging buffer could not be made ({}); the history stands still",
                     stagingFailure);
    }

    hostColour.assign(size_t(w) * h * 3, 0.0f);
    hostResult.assign(size_t(w) * h * 3, 0.0f);
    if (stageMotion.buffer != VK_NULL_HANDLE)
        hostMotion.assign(size_t(mvW) * mvH * 2, 0.0f);

    // The host's own copies of the three staging buffers, byte for byte. Sized from the
    // same footprints, including the row pitch, so the copy across is the whole buffer and
    // not a guess at how wide a row is.
    rawColour.assign(size_t(fp.bytes), 0);
    rawResult.assign(size_t(fp.bytes), 0);
    if (stageMotion.buffer != VK_NULL_HANDLE)
        rawMotion.assign(size_t(mvFootprint.bytes), 0);
    else
        rawMotion.clear();

    footprint = fp;
    width = w;
    height = h;
    format = desc.Format;
    motionFormat = mvFormat;
    mvWidth = mvW;
    mvHeight = mvH;
    bufferFailure.clear();
    return true;
}

// The network, built once and on the first frame that needs it.
//
// A quarter second of device work and three hundred megabytes of weights read back into
// it, which is why it is not built with the buffers: a frame that never reaches the slot
// -- every frame before the first cut, which is the whole of a game's opening menu --
// would pay for a network it never uses.
bool Backend::Impl::EnsureNetwork(UINT w, UINT h)
{
    if (network.Ready())
        return true;
    if (networkTried)
        return false;

    networkTried = true;

    const auto started = std::chrono::steady_clock::now();

    std::string why;
    if (!network.Prepare(w, h, NrFolder(), why))
    {
        networkFailure = why;
        LOG_WARN("NrBridge: the network did not come up ({}): frames go through", why);
        return false;
    }

    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    LOG_INFO("NrBridge: network up in {:.0f} ms on {}x{} (field {}x{}) from {}", ms, w, h, network.NetWidth(),
             network.NetHeight(), NrFolder().string());

    // The submit's three costs, asked for the moment the network is up. The gap's own
    // timings see the whole of `work` as one block; these split it into the CPU handing
    // the graph over, the CPU waiting for the device, and the timestamp read-back --
    // which is what tells a frame slow on the device from a frame slow in the host's
    // waiting (src/gpu/nr_net.h:127-142). Three clock reads a frame is a diagnostic's
    // price, paid while the structure is being measured; the report clears the span, so
    // the gap log reads it every fiftieth frame.
    if (!network.TimeSubmit(true))
        LOG_INFO("NrBridge: this network library does not offer the submit timings");
    return true;
}

Backend::Backend(ID3D12Device* device, ID3D12CommandQueue* queue, const std::filesystem::path& directory)
    : p(new Impl { device, queue, directory })
{
}

// Nothing is recorded yet: this is the seam the renderer will be attached to, and until it is,
// the frame the game handed over is the frame the upscaler gets. A null return is the reference's
// own signal for "skip", so the caller's behaviour is already correct either way.

ID3D12Resource* Backend::Record(ID3D12GraphicsCommandList* cmd, const Frame& f, const Settings&)
{
    if (f.motion == nullptr || f.colour == nullptr)
        return nullptr;

    // The frame's own extent: what the upscaler was told, and what the copy is described with.
    // The parameter block is the authority, and the texture only a fallback for a frame that
    // did not report one.
    const auto desc = f.colour->GetDesc();
    const UINT w = f.width != 0 ? f.width : static_cast<UINT>(desc.Width);
    const UINT h = f.height != 0 ? f.height : static_cast<UINT>(desc.Height);

    // What the host needs of the frame besides its pixels: the exposure the game scaled the
    // scene by, and how far the vectors move a pixel. The reference's own reading of the
    // scale, `motionScaleX = motion_scale_x / motionWidth`, is a shift in uv; the pass wants
    // it in pixels of the frame's extent.
    p->preExposure = f.preExposure > 0.0f ? f.preExposure : 1.0f;
    // A frame that asks for a reset, or the first one, has nothing behind it: the input
    // pass then repeats the colour in the history's channels, which is what the vendor's
    // first-frame layout does (nr_image.c feature_pixel, "was ? ... : scaled").
    if (f.reset)
        p->freshHistory = true;
    p->pixelsPerUvX = 0;
    p->pixelsPerUvY = 0;
    p->haveMotion = false;
    if (p->mvWidth != 0 && p->mvHeight != 0)
    {
        p->pixelsPerUvX = f.motionScaleX / float(p->mvWidth) * float(w);
        p->pixelsPerUvY = f.motionScaleY / float(p->mvHeight) * float(h);
    }

    /* Jittered vectors, and the reference's answer, word for word
     * (`ta_be_MochizukiBackend.cpp:223-224`):
     *
     *   // Jittered vectors cannot be used as the unjittered history flow without conversion.
     *   frame.motion = input.motionJittered ? nullptr : input.motion;
     *
     * It does not compensate for them; it refuses them. Our pass cannot either: `nr_pre.comp`
     * reprojects the history at `uv + mv * scale` with nothing that knows the frame was
     * jittered, so a jittered vector is not a slightly wrong reprojection, it is the
     * history read from the wrong place. And it is jittered here -- the log prints
     * `jitter=(0.375,0.056)`, `(0.438,-0.241)`, `(0.406,0.352)`, moving every frame, and
     * `IFeature.cpp:54` sets `_initFlags.JitteredMV` from `[JitterCancellation]`, which is
     * also what puts `XESS_INIT_FLAG_JITTERED_MV` on XeSS's own init (XeSSFeature.cpp:75).
     *
     * The frame still runs. What it loses is the reprojection, so the network reads the
     * history at this pixel's own place -- `prev_uv == uv`, which is the colour-only
     * behaviour, and is the degradation the reference documents as verified rather than
     * guessed (`notes/motion-source-plan.md`, quoting `motion_estimate.comp`):
     * "where the match cannot be justified we write exactly zero, which makes prev_uv ==
     * uv ... the degradation target is a verified state, not a guess." */
    const bool vectorsUsable = !f.motionJittered;
    if (!vectorsUsable)
    {
        static std::atomic<bool> saidJitter { false };
        if (!saidJitter.exchange(true, std::memory_order_relaxed))
            LOG_INFO("NrBridge: the game's vectors carry jitter (jitter=({:.3f},{:.3f})), so they "
                     "are not given to the network: the history is read at this pixel's own place, "
                     "as the reference does with jittered vectors",
                     f.jitterX, f.jitterY);
    }

    /* Whether this frame can be cut at all, asked before anything is recorded on it --
     * the reference's order and its reason, at `ta_be_MochizukiBackend.cpp:187-192`:
     *
     *   if (FAILED(cmd->QueryInterface(IID_PPV_ARGS(&logical))) || logical->IsSplitIneligible() ||
     *       FAILED(cmd->QueryInterface(IID_PPV_ARGS(&resources))) || !resources->CanAppendCompute())
     *   { p->Status("mochizuki: recording boundary unavailable; original frame"); return nullptr; }
     *
     * We asked after the copies were recorded, which is how twelve percent of every two
     * hundred frames ended up colour-and-motion vectors recorded onto a list that could
     * never be cut: `declined ... ours=false`, on a list the game holds and reuses, once
     * every eight frames or so. Those frames ran nothing and showed the game its own
     * colour, which beside the frames that were denoised is the flicker.
     *
     * It is not a different list each time -- the log shows one address repeated -- so it
     * is not a race either: this list was never ours to begin with, and asking the proxy
     * first is what tells us so while there is still nothing recorded on it. */
    {
        DlssNr::Submission::ILogicalCommandList* logical = nullptr;
        DlssNr::Submission::IRecordingResources* resources = nullptr;
        const bool haveLogical =
            SUCCEEDED(cmd->QueryInterface(__uuidof(DlssNr::Submission::ILogicalCommandList),
                                          reinterpret_cast<void**>(&logical))) && logical != nullptr;
        const bool ineligible = haveLogical && logical->IsSplitIneligible();
        const char* why = haveLogical ? logical->SplitRejectionReason() : "(not our list)";
        const bool haveResources =
            haveLogical && SUCCEEDED(logical->QueryInterface(
                                   __uuidof(DlssNr::Submission::IRecordingResources),
                                   reinterpret_cast<void**>(&resources))) &&
            resources != nullptr;
        const bool canCompute = haveResources && resources->CanAppendCompute();
        if (resources != nullptr)
            resources->Release();
        if (logical != nullptr)
            logical->Release();

        if (!haveLogical || ineligible || !haveResources || !canCompute)
        {
            // Counted by reason. The four are independent questions and they want different fixes:
            // `!haveLogical` is the game's own list and only an earlier hook reaches it;
            // `ineligible` is the proxy's own verdict on a list it did wrap; `!haveResources`
            // is the proxy's bookkeeping being absent; `!canCompute` is `CanAppendCompute()`
            // saying no, which on this host is `rawInterfaceEscaped` -- the proxy leaked a
            // native interface at some point and says no for the rest of its life
            // (`CommandListProxy.h`: "Lifetime-wide; an alias can outlive Reset").
            if (!haveLogical)
                g_declNotOurs.fetch_add(1, std::memory_order_relaxed);
            else if (ineligible)
                g_declIneligible.fetch_add(1, std::memory_order_relaxed);
            else if (!haveResources)
                g_declNoResources.fetch_add(1, std::memory_order_relaxed);
            else
                g_declNoCompute.fetch_add(1, std::memory_order_relaxed);

            const auto no = g_splitNoProxy.fetch_add(1, std::memory_order_relaxed) + 1;
            if (no <= 12)
                LOG_INFO("NrBridge: frame not cuttable: {} -- the original frame is shown, "
                         "as the reference does", why ? why : "(none)");
            g_lastCutHr.store(S_FALSE, std::memory_order_relaxed);
            return nullptr;
        }
    }

    const bool haveBuffers = p->BuildBuffers(f.colour, f.motion, w, h);

    // The pass wants the vectors at the frame's own extent, because that is where it
    // samples the history. At another extent they are not carried at all, and the history
    // stands still, which is a frame the pass can still run on.
    p->haveMotion = vectorsUsable && p->stageMotion.buffer != VK_NULL_HANDLE &&
                    !p->hostMotion.empty() && p->mvWidth == w && p->mvHeight == h;

    if (!haveBuffers)
    {
        // Said once: a frame without buffers is a frame passed through, which is not worth a
        // line per frame, but the reason it has none is worth one line at all.
        static std::atomic<bool> said { false };
        if (!said.exchange(true, std::memory_order_relaxed))
            LOG_WARN("NrBridge: the frame's buffers could not be made ({}): frames are passed through",
                     p->bufferFailure);
    }

    // The reference's RecordInputs, in the reference's place: before the cut, so that it lands
    // in the producer half and is complete before the slot's wait is satisfied. The frame's
    // colour goes into the buffer the slot's copy reads from.
    if (haveBuffers)
    {
        CopyToBuffer(cmd, f.colour, f.colourState, p->input.resource.Get(), p->footprint, w, h);

        // The motion vectors cross in the same half of the cut and into a buffer of their own,
        // for the same reason the colour does: whatever runs in the slot needs both, and either
        // one on its own is a frame the other cannot be matched to. Nothing reads this buffer
        // yet -- the slot's work is still the reference's passthrough -- but it is filled from
        // here on, which is the part that has to be right before anything is asked to use it.
        if (vectorsUsable && p->mvFootprint.valid() && p->mvInput.resource != nullptr)
        {
            CopyToBuffer(cmd, f.motion, f.motionState, p->mvInput.resource.Get(), p->mvFootprint,
                         p->mvWidth, p->mvHeight);

            // Printed for the first few frames only: what the motion copy was described with is
            // what decides whether the vectors arrive whole, and it is worth seeing once.
            if (p->recorded.load(std::memory_order_relaxed) < 3)
                LOG_INFO("NrBridge mv: {}x{} fmt={} pitch={} bytes={}", p->mvWidth, p->mvHeight,
                         static_cast<int>(p->motionFormat), p->mvFootprint.d3d.Footprint.RowPitch,
                         p->mvFootprint.bytes);
        }
    }

    // The cut goes here because that is where the reference puts it, in
    // ta_be_MochizukiBackend.cpp:
    //
    //     if (p->api.RecordInputs(...) != LMXXF_NR_OK ||
    //         LmxxfCut::TrySplitAtEvaluate(cmd) != S_OK ||
    //         p->api.RecordOutputs(...) != LMXXF_NR_OK)
    //
    // between recording the network's inputs and recording its outputs.
    const HRESULT hr = DlssNr::Backend::LmxxfCut::TrySplitAtEvaluate(cmd);
    g_lastCutHr.store(hr, std::memory_order_relaxed);
    if (hr == S_OK)
    {
        g_splitOk.fetch_add(1, std::memory_order_relaxed);

        // Arm the slot for this list, in the same place the reference does it: its
        // LmxxfBackend::Record sets the pending enqueue after the cut, between the calls
        // that record the network's inputs and its outputs. BetweenThunk then calls the
        // function once, when the cut list reaches the queue, between its two halves.
        const auto serial = g_splitOk.load(std::memory_order_relaxed);

        // Logged for the first twenty, for the same reason the gap run is: a crash between
        // the cut and the slot has to be visible in the log, and the periodic line is too
        // sparse to show it.
        if (serial <= 20)
            LOG_INFO("NrBridge cut #{}: list {} armed for the slot", serial, static_cast<void*>(cmd));

        DlssNr::Backend::LmxxfCut::SetPendingEnqueue(this, GapJobHandle(serial), &EnqueueGap, nullptr,
                                                     static_cast<ID3D12CommandList*>(cmd), nullptr);
    }
    else if (hr == S_FALSE)
    {
        /* Not one of our lists, or the list would not split. Which of the two is the
         * difference between a frame that is denoised and a frame that is only the game's
         * own colour, and the reference already answers it: `SplitRejectionReason` is its
         * own word for it, and asking it costs a call. Counted here rather than sampled,
         * because the frames that are missing are exactly the ones a periodic line never
         * lands on -- that is what made this look like a mystery for hours. */
        const auto no = g_splitNoProxy.fetch_add(1, std::memory_order_relaxed) + 1;
        const char* why = "(not our list)";
        if (DlssNr::Submission::ILogicalCommandList* logical = nullptr;
            SUCCEEDED(cmd->QueryInterface(__uuidof(DlssNr::Submission::ILogicalCommandList),
                                          reinterpret_cast<void**>(&logical))) &&
            logical != nullptr)
        {
            const char* said = logical->SplitRejectionReason();
            why = said != nullptr && *said ? said : "(eligible but not split)";
            logical->Release();
        }
        /* Which list it was, and whether it says it is ours. A list that reports itself as
         * the game made it is a list the create hook never saw, and the one thing that
         * explains thirteen percent of frames forever rather than only at the start: the
         * hook is not covering every path the game makes a list on. `AssertMode` and the
         * node mask are in the same line because they are the two other things the create
         * hook refuses on. */
        if (no <= 12)
        {
            /* What the list says about itself. `GetModuleFileNameA` came back empty last
             * time, which only says the address is not inside a loaded module -- a COM
             * object's own code is, but a list that reports itself as the game's means it
             * never went through our create hook at all, and the two want different fixes.
             * The interface name and the device pointer are what separate them. */
            char who[256] = {};
            HMODULE from = nullptr;
            if (from == nullptr &&
                GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   reinterpret_cast<LPCWSTR>(cmd), &from))
                GetModuleFileNameA(from, who, sizeof who - 1);

            ID3D12Device* made = nullptr;
            if (SUCCEEDED(cmd->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void**>(&made))) &&
                made != nullptr)
                made->Release();

            const bool ours = [&] {
                DlssNr::Submission::ILogicalCommandList* probe = nullptr;
                if (FAILED(cmd->QueryInterface(__uuidof(DlssNr::Submission::ILogicalCommandList),
                                               reinterpret_cast<void**>(&probe))) || probe == nullptr)
                    return false;
                probe->Release();
                return true;
            }();

            LOG_INFO("NrBridge: cut #{} declined: {} (hr 0x{:x}) list {} module='{}' device={} "
                     "ours={} armed={} proxyWrap={}",
                     no, why, static_cast<unsigned>(hr), static_cast<void*>(cmd), who,
                     static_cast<void*>(made), ours,
                     DlssNr::Submission::Hooks::IsArmed(),
                     DlssNr::Submission::Hooks::ProxyWrapEnabled());
        }
    }
    else
        g_splitFailed.fetch_add(1, std::memory_order_relaxed);

    // The reference's RecordOutputs, in the reference's place: after the cut, so that it lands
    // in the continuation half, which the game's queue does not reach until the slot's submit
    // has signalled. What comes back out of the buffer is the slot's copy, which for now is
    // the frame's own colour -- the reference's Passthrough, in the network's place.
    //
    // Only for a frame that was actually split. The slot is what fills the buffer this reads
    // back from, and the slot is only run for a list that was split: for one that was not,
    // nothing has written that buffer, and handing the result over would hand over whatever
    // it held before -- which, on the frames before the first successful cut, is nothing at
    // all. The game's opening menu is spent on exactly those frames (a minute of them in the
    // run of 2026-10-13, every one of them S_FALSE: the lists are made before the swapchain
    // is, and that is when the host starts wrapping them), and it came up black for it. A
    // frame that was not split is passed through, which is what the reference's null answer
    // is for.
    if (hr == S_OK && haveBuffers && p->result != nullptr)
    {
        CopyFromBuffer(cmd, p->result.Get(), p->output.resource.Get(), p->footprint, w, h, f.colourState,
                       p->resultState);

        // What the result is left in is not what this module left it in. It is handed to the
        // upscaler in the state the upscaler wants the colour in, and the upscaler then
        // barriers it out of that state and into the one it reads it in, which on this host's
        // Unreal path is NON_PIXEL_SHADER_RESOURCE (XeSSFeature_Dx12.cpp: ColorResourceBarrier
        // is set to RENDER_TARGET for Unreal, and the barrier runs from there). A texture does
        // not decay the way a buffer does, so that -- not the state handed over -- is what the
        // next frame's barrier has to name as its StateBefore. Naming the handed-over state
        // instead put a barrier with a wrong StateBefore in front of every copy after the
        // first, and the opening frames came out black.
        p->resultState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        p->recorded.fetch_add(1, std::memory_order_relaxed);

        // The reference hands the result of Record to the upscaler as its colour input, and
        // leaves the frame alone when it answers null. Same here.
        return p->result.Get();
    }

    p->recorded.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
}

int Backend::PendingListIndex(UINT, ID3D12CommandList* const*) const { return -1; }
void Backend::Submitting(ID3D12CommandQueue*, UINT, ID3D12CommandList* const*) { }
void Backend::TraceBoundary(const std::string&) { }
// The reference drops a pending enqueue once the list it was waiting for has gone to the queue,
// so a job can never be enqueued twice by a later submission of the same list.
void Backend::Submitted(ID3D12CommandQueue*, UINT count, ID3D12CommandList* const* lists)
{
    DlssNr::Backend::LmxxfCut::ClearPendingEnqueueIfSubmitted(count, lists);
}
bool Backend::Ready() { return true; }
bool Backend::Shutdown() { return true; }
void Backend::ReleaseSession() { }
bool Backend::PollRelease() { return true; }
bool Backend::CompletePendingReleaseLocked() { return true; }
void Backend::ResetGraphicsWaitState() { }
void Backend::InvalidateHistory() { }

bool Backend::SlotCopy(VkBufferCopy* region, VkBuffer* from, VkBuffer* to)
{
    if (region == nullptr || from == nullptr || to == nullptr)
        return false;

    if (p->input.buffer == VK_NULL_HANDLE || p->output.buffer == VK_NULL_HANDLE || !p->footprint.valid())
        return false;

    // The whole footprint: the rows the frame's colour was copied into, and no more.
    region->srcOffset = 0;
    region->dstOffset = 0;
    region->size = p->footprint.bytes;
    *from = p->input.buffer;
    *to = p->output.buffer;
    return true;
}

// The three calls that put the network in the slot, in the order the gap makes them.

bool Backend::SlotRead(VulkanGap::Transfer* transfers, UINT& count)
{
    count = 0;
    if (transfers == nullptr)
        return false;

    // The handle is what both paths need; the imported buffer is only for the one
    // that carries the frame by copy.
    if (p->input.handle == nullptr || !p->footprint.valid())
        return false;

    // The frame's own buffers, handed to the network once, before it is opened: after
    // this the recording has the two fused passes in it and no staging is wanted.
    if (!p->network.Fused())
    {
        std::string fusedWhy;
        if (!p->network.Fuse(NrFolder(), reinterpret_cast<unsigned long long>(p->input.handle),
                             p->mvInput.resource != nullptr
                                 ? reinterpret_cast<unsigned long long>(p->mvInput.handle)
                                 : 0ull,
                             reinterpret_cast<unsigned long long>(p->output.handle),
                             // The resource's own size, not the frame's footprint: the same
                             // handle is imported on the device the network runs on, and an
                             // import has to be the object the hand-over's import is (16 MB
                             // floor and 64 KB rounding, SharedBuffer.h). A buffer imported
                             // at the footprint's size is a different object from the one
                             // behind the handle.
                             p->input.size,
                             p->mvInput.resource != nullptr ? p->mvInput.size : 0ull, p->output.size, p->width,
                             p->height, p->footprint.d3d.Footprint.RowPitch / 4u, p->preExposure,
                             fusedWhy))
            LOG_WARN("NrBridge: the fused path would not start: {}", fusedWhy);
    }

    if (!p->EnsureNetwork(p->width, p->height))
        return false;

    // Nothing to stage: the graph reads the game's own buffer and writes its own answer.
    if (p->network.Fused())
        return true;

    // The network did not take the frame, so this side carries it -- and carrying it is
    // the only reason to import the handles here. Left unimported on the fused path, the
    // network's device is the one device that imports them, as in the reference.
    std::string importWhy;
    if (!p->input.Import(importWhy) || !p->output.Import(importWhy) ||
        p->stageColour.buffer == VK_NULL_HANDLE)
        return false;

    transfers[count].from = p->input.buffer;
    transfers[count].to = p->stageColour.buffer;
    transfers[count].region.srcOffset = 0;
    transfers[count].region.dstOffset = 0;
    transfers[count].region.size = p->footprint.bytes;
    count += 1;

    // The motion vectors cross in the same submission: the pass wants the frame's colour
    // and the shift that produced it, and either on its own is a frame the other does not
    // match. Without a buffer for them the pass still runs, and its history stands.
    if (p->haveMotion && p->mvFootprint.valid() && p->stageMotion.buffer != VK_NULL_HANDLE &&
        p->mvInput.Import(importWhy))
    {
        transfers[count].from = p->mvInput.buffer;
        transfers[count].to = p->stageMotion.buffer;
        transfers[count].region.srcOffset = 0;
        transfers[count].region.dstOffset = 0;
        transfers[count].region.size = p->mvFootprint.bytes;
        count += 1;
    }

    return true;
}

bool Backend::SlotWork(UINT64 value)
{
    // The fused path: the graph reads the colour in the game's own buffer and writes the
    // answer back into it, so there is nothing here to decode, nothing to compose and
    // nothing to encode. The graph waits for the game's fence on the device, which is
    // what `value` is for — the same value the game's queue was asked to signal.
    if (p->network.Fused())
    {
        /* The reference's resetPending, consumed. MochiNrRuntime.cpp:2134-2136, on the frame
         * that arrived on a different queue than the last one:
         *
         *   "The new queue's first half has already copied its colour, which may race the
         *    last frame's network: both frames may look wrong, and the history restarts on
         *    the next frame."
         *
         * So the history is dropped at the next frame, not at the switch: the switch frame is
         * already in flight by the time it is known. `freshHistory` is this layer's own name
         * for that state and the graph already reads it, so consuming the flag here is the
         * whole of it -- there is no second mechanism to add. */
#if 0
        // 🔒 2026-10-13 停用：resetPending 恒为 false（本游戏不换队列），这段是死代码。
        //    复用时要与 Follow()、Gap 的 producer 字段、下面非 fused 的那一处一起打开。
        if (TheGap().resetPending.exchange(false, std::memory_order_relaxed))
        {
            p->freshHistory = true;
            static std::atomic<unsigned long long> saidResets { 0 };
            const auto n = saidResets.fetch_add(1, std::memory_order_relaxed) + 1;
            if (n <= 12)
                LOG_INFO("NrBridge: the history restarts ({}), as the reference does when the frame's queue changed",
                         n);
        }
#endif

        // The frame's own pre-exposure, before the graph is run. It is the frame's and
        // not the open's: this game moves it between a quarter and all of it, and a
        // divisor baked in when the network was opened over- or under-exposes every
        // frame the game has moved away from since.
        p->network.Exposure(p->preExposure, p->freshHistory);

        // Whether the edit is blended in, asked every frame so that flipping it in the
        // menu takes effect on the next one without anything being rebuilt. Zero is the
        // frame's own colour exactly and the graph still runs, which is the whole point:
        // both sides of the comparison cost the same and show the same frame.
        p->network.ApplyModel(Config::Instance()->NrApplyModel.value_or_default());


        p->freshHistory = false;   // every frame after this one has the last one behind it

        const HANDLE fence = VulkanGap::TheBridge().toVkShared;
        std::string why;
        if (fence != nullptr &&
            p->network.RunFused(reinterpret_cast<unsigned long long>(fence), value, why))
            return true;

        LOG_WARN("NrBridge: the fused pass did not run: {}", why);
        return false;
    }

    const UINT w = p->width, h = p->height;
    if (w == 0 || h == 0 || !p->network.Ready())
        return false;

    if (p->stageColour.mapped == nullptr || p->hostColour.empty() || p->hostResult.empty())
        return false;
    if (p->rawColour.size() != size_t(p->footprint.bytes) || p->rawResult.size() != size_t(p->footprint.bytes))
        return false;

    // Across in one copy each, before anything looks at a pixel. This is the whole reason
    // the host has buffers of its own: one copy of four megabytes is a few milliseconds,
    // where the same bytes read a pixel at a time through BAR cost a round trip each.
    const auto tCopyIn = Clock::now();
    std::memcpy(p->rawColour.data(), p->stageColour.mapped, size_t(p->footprint.bytes));
    if (p->haveMotion && p->stageMotion.mapped != nullptr && p->rawMotion.size() == size_t(p->mvFootprint.bytes))
        std::memcpy(p->rawMotion.data(), p->stageMotion.mapped, size_t(p->mvFootprint.bytes));

    const auto tDecode = Clock::now();

    // The colour, out of the format the game left it in and out of the exposure it was
    // scaled by, and clipped to what the network was trained on.
    NrBridge::DecodeColour(p->rawColour.data(), w, h, p->footprint.d3d.Footprint.RowPitch, p->preExposure,
                           p->hostColour.data(), &p->lastLow, &p->lastHigh);

    const auto tMotion = Clock::now();

    // The vectors, in pixels. Not finite is not a shift, and a shift of a whole screen is
    // not one either: both are a pixel that could not be tracked.
    std::vector<float>* motion = nullptr;
    if (p->haveMotion && p->stageMotion.mapped != nullptr && !p->hostMotion.empty() && !p->rawMotion.empty())
    {
        const uint8_t* rows = p->rawMotion.data();
        const size_t pitch = p->mvFootprint.d3d.Footprint.RowPitch;

        for (UINT y = 0; y < p->mvHeight; ++y)
        {
            const uint16_t* line = reinterpret_cast<const uint16_t*>(rows + y * pitch);
            for (UINT x = 0; x < p->mvWidth; ++x)
            {
                const size_t at = (size_t(y) * p->mvWidth + x) * 2;
                float u = NrBridge::HalfValue(line[x * 2 + 0]);
                float v = NrBridge::HalfValue(line[x * 2 + 1]);

                if (!std::isfinite(u))
                    u = 0.0f;
                if (!std::isfinite(v))
                    v = 0.0f;

                p->hostMotion[at + 0] = std::clamp(u * p->pixelsPerUvX, -float(w), float(w));
                p->hostMotion[at + 1] = std::clamp(v * p->pixelsPerUvY, -float(h), float(h));
            }
        }
        motion = &p->hostMotion;
    }

    const auto tNet = std::chrono::steady_clock::now();

    // The same resetPending the fused path consumes, on the path that has a history buffer
    // rather than a flag. ResetHistory seeds `previous_` from this frame's own colour and
    // clears haveHistory_, which is exactly "the history restarts".
#if 0
    // 🔒 2026-10-13 停用：同 fused 路径那一处，随 Follow() 一起。
    if (TheGap().resetPending.exchange(false, std::memory_order_relaxed))
        p->network.RestartHistory(p->hostColour.data());
#endif

    std::string why;
    if (!p->network.Run(p->hostColour.data(), motion != nullptr ? motion->data() : nullptr, p->hostResult.data(),
                        ReprojectHistory(), why))
    {
        static std::atomic<unsigned long long> said { 0 };
        if (said.fetch_add(1, std::memory_order_relaxed) == 0)
            LOG_WARN("NrBridge: the pass did not run ({}): frames go through", why);
        return false;
    }

    p->lastFeaturesMs = p->network.FeaturesMs();
    p->lastNetMs = p->network.NetMs();
    p->lastComposeMs = p->network.ComposeMs();
    p->lastGate = p->network.GateMean();
    p->network.HeadStats(p->lastHeadMean, p->lastHeadPeak);

    // Back into the game's format, with the exposure put back: the upscaler is handed what
    // the game would have handed it, scaled the way the game scales it. Written into the
    // host's own buffer, and taken across in one copy underneath: the same reason as on the
    // way in.
    const auto tEncode = std::chrono::steady_clock::now();
    NrBridge::EncodeColour(p->hostResult.data(), w, h, p->footprint.d3d.Footprint.RowPitch, p->preExposure,
                           p->rawResult.data());

    const auto tCopyOut = std::chrono::steady_clock::now();
    std::memcpy(p->stageResult.mapped, p->rawResult.data(), size_t(p->footprint.bytes));

    const auto tEnd = std::chrono::steady_clock::now();
    p->lastCopyMs = ElapsedMs(tCopyIn, tDecode) + ElapsedMs(tCopyOut, tEnd);
    p->lastDecodeMs = ElapsedMs(tDecode, tMotion);
    p->lastMotionMs = ElapsedMs(tMotion, tNet);
    p->lastEncodeMs = ElapsedMs(tEncode, tEnd);
    return true;
}

bool Backend::SlotWrite(VulkanGap::Transfer* transfers, UINT& count)
{
    count = 0;
    if (transfers == nullptr)
        return false;

    /* The fused path first, and before any question about staging. It was last, behind a
     * check for the staging buffers that a fused frame deliberately does not have -- the
     * answer is in the game's own buffer, so there is nothing to stage -- so the check
     * could not pass, this answered false, and the caller fell through to its passthrough:
     * a frame the network had just denoised, overwritten with the game's own colour. The
     * log said it plainly and it was not read: `copies=0 nowork=2008` with `work=69ms`
     * beside it, which is a network running and its answer being thrown away every frame.
     * That is the flicker. */
    if (p->network.Fused())
        return true;

    if (p->stageResult.buffer == VK_NULL_HANDLE || p->output.buffer == VK_NULL_HANDLE || !p->footprint.valid())
        return false;

    transfers[count].from = p->stageResult.buffer;
    transfers[count].to = p->output.buffer;
    transfers[count].region.srcOffset = 0;
    transfers[count].region.dstOffset = 0;
    transfers[count].region.size = p->footprint.bytes;
    count += 1;
    return true;
}

bool Backend::NetworkDevice(DeviceHandles* out)
{
    if (out == nullptr)
        return false;
    if (!Config::Instance()->NrNetwork.value_or_default())
        return false;
    return p->network.Boot(NrFolder(), out);
}

bool Backend::NetworkUp() const { return p->network.Ready(); }
const char* Backend::NetworkFailure() const { return p->networkFailure.c_str(); }
double Backend::LastFeaturesMs() const { return p->lastFeaturesMs; }
double Backend::LastNetMs() const { return p->lastNetMs; }
double Backend::LastComposeMs() const { return p->lastComposeMs; }
bool Backend::SubmitTimings(double out[12]) { return p->network.TimeReport(out); }
double Backend::LastDecodeMs() const { return p->lastDecodeMs; }
double Backend::LastMotionMs() const { return p->lastMotionMs; }
double Backend::LastEncodeMs() const { return p->lastEncodeMs; }
double Backend::LastCopyMs() const { return p->lastCopyMs; }
float Backend::LastGate() const { return p->lastGate; }
float Backend::LastHeadMean() const { return p->lastHeadMean; }
float Backend::LastHeadPeak() const { return p->lastHeadPeak; }
float Backend::LastLow() const { return p->lastLow; }
float Backend::LastHigh() const { return p->lastHigh; }
std::string Backend::Status() const { return "no renderer attached"; }
bool Backend::IsRunning() const { return false; }
UINT64 Backend::RecordedFrames() const { return p->recorded.load(std::memory_order_relaxed); }

/* The hook, at the device's own creation, with no queue and nothing else running.
 *
 * The reference has this entry for exactly this reason and names it ArmCreate
 * (`SubmissionHooks.h`, taken as it was written): the create hook needs a device and
 * nothing else, so it can be in place before a game has made anything at all -- where
 * `EnsureSubmissionArmed` cannot, because it waits for a queue and a frame.
 *
 * The log is what says this game needs it: `declined=1813` against `cutsRecorded=586`,
 * every decline "(not our list)". Three evaluates in four arrive with a list the proxy
 * never wrapped, so three frames in four are the game's own colour -- the flicker. The
 * reference's own note above that switch says who does this: "UE can retain lists created
 * before the first swapchain."
 *
 * Calling it early is harmless if it is too late: ArmCreate answers S_FALSE once armed,
 * and EnsureSubmissionArmed's own path is unchanged. */
void ArmSubmissionAtDeviceCreation(ID3D12Device* device)
{
    if (device == nullptr)
        return;
    if (!Config::Instance()->NrWrapLists.value_or_default())
        return;

    /* Order here is the reference's, and each line is a rule it states beside the code
     * (`ta_D3D12_Hooks.cpp`, the block after `ArmCreate`):
     *
     *   SetWrapOpenLists(...)  open-list proxy is on in the product path.
     *   ArmCreate(device)      Create* detours only, at the device's own creation.
     *   ProxyWrap stays OFF    "wrapping every DIRECT list during Streamline/device boot
     *                           crashes yysls" -- it is the swapchain's job.
     *   allowEarly             whitelist: Unreal/Forza host calls only; other engines
     *                           crash with early ArmCreate + wrap.
     *   earlyQueue + ready     the gate. Early wrapping is on only if the queue detour
     *                           was installed, because the between slot needs it.
     *
     * The earlier attempt turned ProxyWrap and WrapOpenLists on here as well; that is the
     * line the reference names as the crash, and it lost the device at
     * `SwapChain1->ResizeBuffers`. */
    DlssNr::Submission::Hooks::SetProxyWrap(false);
    DlssNr::Submission::Hooks::SetWrapOpenLists(true);
    const auto hr = DlssNr::Submission::Hooks::ArmCreate(device);
    if (FAILED(hr))
    {
        LOG_INFO("NrBridge: create hook not installed at device creation (0x{:x})", static_cast<unsigned>(hr));
        return;
    }
    LOG_INFO("NrBridge: the create hook is in place at the device's own creation, so the lists the game "
             "makes from here are ours; ProxyWrap stays off until the swapchain");

    const auto& st = State::Instance();
    std::string exeLower = st.GameExe;
    for (char& c : exeLower)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const bool isUnreal = st.GameEngine == GameEngineType::Unreal;
    // LOP-Win64-Shipping.exe: the reference's whitelist is engine-based, and the engine
    // here is only half detected at this point (the version-info check runs later), so
    // the shipping-exe shape it also accepts is accepted too.
    const bool isUnrealShape = exeLower.ends_with("-win64-shipping.exe") || exeLower.ends_with("-wingdk-shipping.exe") ||
                               exeLower.ends_with("-windows-shipping.exe");
    const bool allowEarly = isUnreal || isUnrealShape;
    LOG_INFO("NrBridge: early exe wrap allow={} (unreal={} shape={})", allowEarly, isUnreal, isUnrealShape);

    if (!allowEarly)
    {
        DlssNr::Submission::Hooks::SetEarlyExeWrap(false);
        return;
    }

    // The gate, asked on a temporary queue exactly as the reference asks it: no game
    // queue exists yet this early, and a DIRECT queue is all the detour needs.
    D3D12_COMMAND_QUEUE_DESC queueDesc {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ID3D12CommandQueue* earlyQueue = nullptr;
    const bool created = SUCCEEDED(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&earlyQueue)));
    const bool ready = created && DlssNr::Submission::Hooks::EnsureExecuteHook(earlyQueue);
    if (earlyQueue != nullptr)
        earlyQueue->Release();
    DlssNr::Submission::Hooks::SetEarlyExeWrap(allowEarly && ready);
    LOG_INFO("NrBridge: early executable proxy: {} (submission hook ready={})", allowEarly && ready, ready);
}

void EvaluateBeforeUpscale(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params,
                           ID3D12CommandQueue* timingQueue)
{
    if (params == nullptr)
        return;

    const auto n = g_seen.fetch_add(1, std::memory_order_relaxed);

    if (!Config::Instance()->NrEnabled.value_or_default())
        return;

    if (g_failed.load(std::memory_order_acquire))
        return;

    const Frame f = ReadFrame(params);
    LogFrame(f);

    // Asked before anything is recorded, and answered by the list itself rather than assumed:
    // a game that evaluates inside a render pass cannot be cut, and finding that out here costs
    // one log line instead of a hung device.
    if (cmdList != nullptr)
        ReportSplitEligibility(cmdList, n);

    if (!Config::Instance()->NrRunBeforeSR.value_or_default())
        return;

    auto* backend = EnsureBackend(timingQueue);
    if (backend == nullptr)
        return;

    Settings settings {};
    auto* input = backend->Record(cmdList, f, settings);

    // The reference hands the result of Record to the upscaler as its colour input, and leaves
    // the frame alone when it answers null. Same here.
    if (input != nullptr && input != f.colour)
        params->Set(NVSDK_NGX_Parameter_Color, static_cast<void*>(input));

    if (n == 5)
        LOG_INFO("NrBridge: recorded {} frames", backend->RecordedFrames());
}

void Shutdown()
{
    auto* backend = g_backend.exchange(nullptr, std::memory_order_acq_rel);
    if (backend != nullptr)
        backend->Shutdown();
}

bool IsRunning()
{
    auto* backend = g_backend.load(std::memory_order_acquire);
    return backend != nullptr && backend->IsRunning();
}

const char* FailureReason() { return g_failureReason.c_str(); }

void RetryAfterFailure()
{
    g_failureReason.clear();
    g_failed.store(false, std::memory_order_release);
}

void RenderMenu(::Config*, float) { }
} // namespace NrBridge
