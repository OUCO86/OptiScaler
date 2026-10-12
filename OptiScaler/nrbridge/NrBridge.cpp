#include "pch.h"

#include <nrbridge/NrBridge.h>

#include "Config.h"
#include "State.h"
#include "Util.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

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
// What goes in now is a probe, and it is deliberately the least a probe can be: one command list
// of our own on the queue the game is already submitting to, copying a counter out of an upload
// buffer we own into a buffer we own. It reads nothing the game made and writes nothing the game
// will read, so there is no resource state to guess and no way for it to change the picture. The
// question it answers is only whether the slot runs at all, and what a submission there costs.
constexpr UINT GapSlotCount = 3;
constexpr UINT64 GapPayloadBytes = 256;

// Stands in for the reference's job handle until there is a job. Its own handles are odd, so
// never null and never a pointer, and BetweenThunk refuses to call in with a null session or
// job -- it counts a skip and returns without running anything.
void* GapJobHandle(unsigned long long serial)
{
    return reinterpret_cast<void*>(static_cast<uintptr_t>((serial << 1) | 1));
}

struct Gap
{
    std::mutex mutex;
    Microsoft::WRL::ComPtr<ID3D12Device> device;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocators[GapSlotCount];
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> lists[GapSlotCount];
    Microsoft::WRL::ComPtr<ID3D12Resource> staging;
    Microsoft::WRL::ComPtr<ID3D12Resource> target;
    void* mapped = nullptr;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    UINT64 signalled[GapSlotCount] {};
    UINT64 nextFence = 1;
    unsigned next = 0;
    bool ready = false;
    const char* failure = "";

    std::atomic<unsigned long long> runs { 0 };
    std::atomic<unsigned long long> submits { 0 };
    std::atomic<unsigned long long> failures { 0 };
    std::atomic<double> cpuTotalMs { 0.0 };
    std::atomic<double> cpuMaxMs { 0.0 };
};

Gap& TheGap()
{
    // Process lifetime owner, the same way the reference keeps its Pending(): it is not torn
    // down while the game still has threads recording.
    static Gap g;
    return g;
}

bool BuildGap(ID3D12Device* device)
{
    Gap& g = TheGap();

    for (UINT i = 0; i < GapSlotCount; ++i)
    {
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                  IID_PPV_ARGS(&g.allocators[i]))))
        {
            g.failure = "CreateCommandAllocator";
            return false;
        }
    }

    // Created the ordinary way, not through the reference's own CreateProxiedCommandList: the
    // lists the game makes are wrapped so that they can be cut, and this one must not be.
    for (UINT i = 0; i < GapSlotCount; ++i)
    {
        if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g.allocators[i].Get(), nullptr,
                                             IID_PPV_ARGS(&g.lists[i]))))
        {
            g.failure = "CreateCommandList";
            return false;
        }
        g.lists[i]->Close();
    }

    D3D12_HEAP_PROPERTIES uploadHeap {};
    uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_HEAP_PROPERTIES defaultHeap {};
    defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC payload {};
    payload.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    payload.Width = GapPayloadBytes;
    payload.Height = 1;
    payload.DepthOrArraySize = 1;
    payload.MipLevels = 1;
    payload.SampleDesc.Count = 1;
    payload.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    // The staging buffer is mapped once and written from the CPU every frame. An upload heap
    // is what makes that legal without a copy of its own.
    if (FAILED(device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &payload,
                                               D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                               IID_PPV_ARGS(&g.staging))))
    {
        g.failure = "CreateCommittedResource(staging)";
        return false;
    }

    if (FAILED(g.staging->Map(0, nullptr, &g.mapped)))
    {
        g.failure = "Map(staging)";
        return false;
    }

    // The target is created in COPY_DEST and never leaves it, so no barrier has to be recorded
    // for it and no state has to be known. That is the whole reason the probe is shaped this way.
    if (FAILED(device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &payload,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                               IID_PPV_ARGS(&g.target))))
    {
        g.failure = "CreateCommittedResource(target)";
        return false;
    }

    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g.fence))))
    {
        g.failure = "CreateFence";
        return false;
    }

    g.ready = true;
    return true;
}

// The reference's EnqueueHipFn, called by BetweenThunk between the two halves.
int32_t EnqueueGap(void*, void* job, void* command_queue)
{
    if (job == nullptr || command_queue == nullptr)
        return -1;

    Gap& g = TheGap();
    const auto started = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lock(g.mutex);

    if (g.failure[0] != 0)
    {
        g.failures.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }

    if (!g.ready)
    {
        auto* queue = static_cast<ID3D12CommandQueue*>(command_queue);
        if (queue == nullptr || FAILED(queue->GetDevice(IID_PPV_ARGS(&g.device))))
        {
            g.failure = "GetDevice";
            g.failures.fetch_add(1, std::memory_order_relaxed);
            return -1;
        }
        if (!BuildGap(g.device.Get()))
        {
            g.failures.fetch_add(1, std::memory_order_relaxed);
            return -1;
        }
    }

    g.runs.fetch_add(1, std::memory_order_relaxed);

    const unsigned slot = g.next++ % GapSlotCount;

    // The allocator cannot be reset while the GPU still has the last frame's copy queued on it,
    // which is why there is a ring and a fence: three deep, the wait is normally already past.
    if (g.signalled[slot] != 0 && g.fence->GetCompletedValue() < g.signalled[slot])
    {
        const HANDLE wait = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (wait != nullptr)
        {
            g.fence->SetEventOnCompletion(g.signalled[slot], wait);
            WaitForSingleObject(wait, 2000);
            CloseHandle(wait);
        }
    }

    auto* allocator = g.allocators[slot].Get();
    auto* list = g.lists[slot].Get();

    if (FAILED(allocator->Reset()))
    {
        g.failures.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }

    if (FAILED(list->Reset(allocator, nullptr)))
    {
        g.failures.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }

    const unsigned long long serial = g.runs.load(std::memory_order_relaxed);
    std::memcpy(g.mapped, &serial, sizeof(serial));
    list->CopyBufferRegion(g.target.Get(), 0, g.staging.Get(), 0, GapPayloadBytes);

    if (FAILED(list->Close()))
    {
        g.failures.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }

    ID3D12CommandList* batch[] = { list };

    // Submitted through the trampoline the reference keeps for exactly this: the game's Execute
    // is the call that got us here, and going back through the hook would expand the list that
    // is being expanded.
    auto* raw = DlssNr::Submission::g_rawExecuteCommandLists;
    if (raw != nullptr)
        raw(static_cast<ID3D12CommandQueue*>(command_queue), 1, batch);
    else
        static_cast<ID3D12CommandQueue*>(command_queue)->ExecuteCommandLists(1, batch);

    const UINT64 signal = g.nextFence++;
    static_cast<ID3D12CommandQueue*>(command_queue)->Signal(g.fence.Get(), signal);
    g.signalled[slot] = signal;

    g.submits.fetch_add(1, std::memory_order_relaxed);

    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    g.cpuTotalMs.store(g.cpuTotalMs.load(std::memory_order_relaxed) + ms, std::memory_order_relaxed);
    double peak = g.cpuMaxMs.load(std::memory_order_relaxed);
    while (ms > peak && !g.cpuMaxMs.compare_exchange_weak(peak, ms, std::memory_order_relaxed))
    {
    }

    return 0;
}

void ReportSplitEligibility(ID3D12GraphicsCommandList* cmdList, unsigned long long n)
{
    const bool early = n < 6;
    const bool periodic = (n % 600) == 0;
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
    const bool periodic = (n % 600) == 0;

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
    LOG_INFO("NrBridge boundary: eval={} cutHr={:X} proxyHits={} cutsRecorded={} rejected={} "
             "unsplitSubmitted={} producerSubmitted={} continuationSubmitted={} submitFailures={} "
             "(counts are NOT GPU completion)",
             n, static_cast<unsigned>(g_lastCutHr.load(std::memory_order_relaxed)),
             g_splitOk.load(std::memory_order_relaxed) + g_splitFailed.load(std::memory_order_relaxed),
             g_splitOk.load(std::memory_order_relaxed), g_splitFailed.load(std::memory_order_relaxed),
             DlssNr::Submission::g_unsplitProxySubmissions.load(std::memory_order_relaxed),
             DlssNr::Submission::g_splitSubmissions.load(std::memory_order_relaxed),
             DlssNr::Submission::g_continuationSubmissions.load(std::memory_order_relaxed),
             DlssNr::Submission::g_submissionFailures.load(std::memory_order_relaxed));

    // Whether the slot runs at all. Its own counts come from the reference's Pending(), which
    // BetweenThunk fills in: betweenHits for a run whose list matched, enqueueCalls for the
    // call into the function, skippedHits for a run that found nothing to do.
    auto& pending = DlssNr::Backend::LmxxfCut::Pending();
    Gap& gap = TheGap();
    LOG_INFO("NrBridge gap: between={} enqueue={} skipped={} | runs={} submits={} failures={} cpu={:.3f}ms "
             "max={:.3f}ms{}",
             pending.betweenHits.load(std::memory_order_relaxed),
             pending.enqueueCalls.load(std::memory_order_relaxed),
             pending.skippedHits.load(std::memory_order_relaxed),
             gap.runs.load(std::memory_order_relaxed), gap.submits.load(std::memory_order_relaxed),
             gap.failures.load(std::memory_order_relaxed), gap.cpuTotalMs.load(std::memory_order_relaxed),
             gap.cpuMaxMs.load(std::memory_order_relaxed), gap.failure[0] != 0 ? gap.failure : "");
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
};

Backend::Backend(ID3D12Device* device, ID3D12CommandQueue* queue, const std::filesystem::path& directory)
    : p(new Impl { device, queue, directory })
{
}

// Nothing is recorded yet: this is the seam the renderer will be attached to, and until it is,
// the frame the game handed over is the frame the upscaler gets. A null return is the reference's
// own signal for "skip", so the caller's behaviour is already correct either way.
ID3D12Resource* Backend::Record(ID3D12GraphicsCommandList* cmd, const Frame& f, const Settings&)
{
    if (f.motion == nullptr)
        return nullptr;

    // The cut goes here because that is where the reference puts it, in
    // ta_be_MochizukiBackend.cpp:
    //
    //     if (p->api.RecordInputs(...) != LMXXF_NR_OK ||
    //         LmxxfCut::TrySplitAtEvaluate(cmd) != S_OK ||
    //         p->api.RecordOutputs(...) != LMXXF_NR_OK)
    //
    // between recording the network's inputs and recording its outputs. There is no network
    // attached yet, so the two calls around it do not exist and nothing is enqueued for the
    // slot to run: what is left is the cut on its own, which is the only thing being measured.
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
        DlssNr::Backend::LmxxfCut::SetPendingEnqueue(this, GapJobHandle(serial), &EnqueueGap, nullptr,
                                                     static_cast<ID3D12CommandList*>(cmd), nullptr);
    }
    else if (hr == S_FALSE)
        g_splitNoProxy.fetch_add(1, std::memory_order_relaxed);
    else
        g_splitFailed.fetch_add(1, std::memory_order_relaxed);

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
std::string Backend::Status() const { return "no renderer attached"; }
bool Backend::IsRunning() const { return false; }
UINT64 Backend::RecordedFrames() const { return p->recorded.load(std::memory_order_relaxed); }

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
