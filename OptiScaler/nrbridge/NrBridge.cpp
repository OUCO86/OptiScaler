#include "pch.h"

#include <nrbridge/NrBridge.h>

#include "State.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>

// Reading the parameter block is deliberately done the way the rest of the host does it: a typed
// Get first, then the untyped one. The host sets these keys from two different places (the FFX
// path and the NGX path) and they do not agree on which overload is populated, so a single Get
// would silently read nothing on one of them.
namespace
{
const char* FormatName(DXGI_FORMAT f)
{
    switch (f)
    {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
        return "R8G8B8A8_UNORM";
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        return "R8G8B8A8_UNORM_SRGB";
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
        return "R16G16B16A16_FLOAT";
    case DXGI_FORMAT_R10G10B10A2_UNORM:
        return "R10G10B10A2_UNORM";
    case DXGI_FORMAT_R11G11B10_FLOAT:
        return "R11G11B10_FLOAT";
    case DXGI_FORMAT_R16G16_FLOAT:
        return "R16G16_FLOAT";
    case DXGI_FORMAT_R16G16_UNORM:
        return "R16G16_UNORM";
    case DXGI_FORMAT_R32_FLOAT:
        return "R32_FLOAT";
    case DXGI_FORMAT_D32_FLOAT:
        return "D32_FLOAT";
    case DXGI_FORMAT_R24G8_TYPELESS:
        return "R24G8_TYPELESS";
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

// IEEE half, which is what R16G16_FLOAT carries. Written out rather than pulled from a
// header because the only thing it is used for is printing four numbers once.
float HalfToFloat(unsigned short h)
{
    const unsigned int sign = static_cast<unsigned int>(h & 0x8000u) << 16;
    const unsigned int mant = h & 0x3FFu;
    const int exp = (h >> 10) & 0x1Fu;
    unsigned int bits;

    if (exp == 0)
    {
        if (mant == 0)
        {
            bits = sign;
        }
        else
        {
            int e = -14;
            unsigned int m = mant;
            while ((m & 0x400u) == 0)
            {
                m <<= 1;
                --e;
            }
            m &= 0x3FFu;
            bits = sign | (static_cast<unsigned int>(e + 127) << 23) | (m << 13);
        }
    }
    else if (exp == 31)
    {
        bits = sign | 0x7F800000u | (mant << 13);
    }
    else
    {
        bits = sign | (static_cast<unsigned int>(exp - 15 + 127) << 23) | (mant << 13);
    }

    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(bits));
    return f;
}

std::atomic<unsigned long long> g_seen { 0 };
std::atomic<bool> g_contractLogged { false };

std::atomic<int> g_samplesLeft { 3 };
std::atomic<bool> g_sampleDead { false };
std::atomic<bool> g_fenceReady { false };
ID3D12Fence* g_fence = nullptr;
HANDLE g_fenceEvent = nullptr;
unsigned long long g_fenceValue = 0;
} // namespace

namespace NrBridge
{
void ProbeBeforeUpscale(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params)
{
    if (params == nullptr)
        return;

    const auto n = g_seen.fetch_add(1, std::memory_order_relaxed);

    Frame f {};
    f.colour = ReadResource(params, NVSDK_NGX_Parameter_Color);
    f.motion = ReadResource(params, NVSDK_NGX_Parameter_MotionVectors);
    f.depth = ReadResource(params, NVSDK_NGX_Parameter_Depth);
    f.exposure = ReadResource(params, NVSDK_NGX_Parameter_ExposureTexture);

    f.hasColour = f.colour != nullptr;
    f.hasMotion = f.motion != nullptr;
    f.hasDepth = f.depth != nullptr;
    f.hasExposure = f.exposure != nullptr;

    f.width = ReadUInt(params, NVSDK_NGX_Parameter_Width);
    f.height = ReadUInt(params, NVSDK_NGX_Parameter_Height);
    f.motionScaleX = ReadFloat(params, NVSDK_NGX_Parameter_MV_Scale_X, 1.0f);
    f.motionScaleY = ReadFloat(params, NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);
    f.jitterX = ReadFloat(params, NVSDK_NGX_Parameter_Jitter_Offset_X, 0.0f);
    f.jitterY = ReadFloat(params, NVSDK_NGX_Parameter_Jitter_Offset_Y, 0.0f);
    f.preExposure = ReadFloat(params, NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1.0f);
    f.reset = ReadUInt(params, NVSDK_NGX_Parameter_Reset) != 0;

    // The first frames are printed in full and then it goes quiet: a per-frame log line at 100
    // frames a second is unreadable, and what matters about the steady state is only whether it
    // is still the same contract.
    const bool early = n < 6;
    const bool periodic = (n % 600) == 0;

    if (!early && !periodic)
        return;

    LOG_INFO("NrBridge frame {} cmdList={} {}x{} mv=({},{}) jitter=({:.3f},{:.3f}) preExp={:.3f} "
             "reset={} | {} | {} | {}",
             n, static_cast<void*>(cmdList), f.width, f.height, f.motionScaleX, f.motionScaleY, f.jitterX,
             f.jitterY, f.preExposure, f.reset ? 1 : 0, Describe("color", f.colour), Describe("mv", f.motion),
             Describe("depth", f.depth));

    if (!g_contractLogged.exchange(true, std::memory_order_relaxed))
    {
        // Printed once, because it is the sentence this whole host exists for. If this says the
        // vectors are absent, nothing downstream is allowed to proceed.
        LOG_INFO("NrBridge contract: motion={} depth={} exposure={} backend={}", f.hasMotion, f.hasDepth,
                 f.hasExposure, Config::Instance()->Dx12Upscaler.value_or_default());

        if (!f.hasMotion)
            LOG_WARN("NrBridge: no motion vectors in this evaluate -- temporal work must fall back");
    }
}

void SampleMotionOnce(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params)
{
    if (g_sampleDead.load(std::memory_order_relaxed))
        return;
    if (g_samplesLeft.fetch_sub(1, std::memory_order_relaxed) <= 0)
        return;

    // A handle that is present is not a texture that holds numbers. Everything after this
    // point assumes the vectors are real, measured, and on the scale the parameter block
    // claims, so the assumption is checked once, on the game's own command list, and the
    // result is printed rather than trusted.
    auto* queue = State::Instance().currentCommandQueue;
    if (queue == nullptr)
    {
        LOG_INFO("NrBridge sample: no command queue yet, skipping (nothing is concluded)");
        return;
    }

    auto* mv = ReadResource(params, NVSDK_NGX_Parameter_MotionVectors);
    if (mv == nullptr)
    {
        g_sampleDead.store(true, std::memory_order_relaxed);
        LOG_WARN("NrBridge sample: motion resource absent -- sampling off");
        return;
    }

    ID3D12Device* device = nullptr;
    if (FAILED(queue->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void**>(&device))) || device == nullptr)
    {
        g_sampleDead.store(true, std::memory_order_relaxed);
        LOG_WARN("NrBridge sample: no device from the queue -- sampling off");
        return;
    }

    auto desc = mv->GetDesc();

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT foot {};
    UINT rows = 0;
    UINT64 rowBytes = 0, total = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &foot, &rows, &rowBytes, &total);

    D3D12_HEAP_PROPERTIES heapProps {};
    heapProps.Type = D3D12_HEAP_TYPE_READBACK;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

    D3D12_RESOURCE_DESC bufDesc {};
    bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufDesc.Width = total;
    bufDesc.Height = 1;
    bufDesc.DepthOrArraySize = 1;
    bufDesc.MipLevels = 1;
    bufDesc.SampleDesc.Count = 1;
    bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ID3D12Resource* readback = nullptr;
    if (FAILED(device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &bufDesc,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                               __uuidof(ID3D12Resource), reinterpret_cast<void**>(&readback))) ||
        readback == nullptr)
    {
        device->Release();
        g_sampleDead.store(true, std::memory_order_relaxed);
        LOG_WARN("NrBridge sample: readback allocation failed -- sampling off");
        return;
    }

    if (!g_fenceReady.exchange(true, std::memory_order_relaxed))
    {
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence),
                                       reinterpret_cast<void**>(&g_fence))) ||
            g_fence == nullptr)
        {
            readback->Release();
            device->Release();
            g_sampleDead.store(true, std::memory_order_relaxed);
            LOG_WARN("NrBridge sample: fence creation failed -- sampling off");
            return;
        }
        g_fenceEvent = CreateEventA(nullptr, FALSE, FALSE, nullptr);
    }

    // The state the host leaves this resource in before an evaluate. Recorded as an
    // assumption: if it is wrong the copy is wrong, and the printed numbers will say so by
    // being nonsense. It is restored immediately afterwards either way.
    const auto assumed = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

    D3D12_RESOURCE_BARRIER in {};
    in.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    in.Transition.pResource = mv;
    in.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    in.Transition.StateBefore = assumed;
    in.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;

    D3D12_RESOURCE_BARRIER out = in;
    out.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    out.Transition.StateAfter = assumed;

    cmdList->ResourceBarrier(1, &in);

    D3D12_TEXTURE_COPY_LOCATION dst {};
    dst.pResource = readback;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = foot;

    D3D12_TEXTURE_COPY_LOCATION src {};
    src.pResource = mv;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = 0;

    cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    cmdList->ResourceBarrier(1, &out);

    const auto value = ++g_fenceValue;
    if (FAILED(queue->Signal(g_fence, value)))
    {
        readback->Release();
        device->Release();
        g_sampleDead.store(true, std::memory_order_relaxed);
        LOG_WARN("NrBridge sample: queue signal failed -- sampling off");
        return;
    }

    // Bounded on purpose. If this command list is not executed on the queue the host happens
    // to have recorded, waiting forever would hang the game, and a missing number never
    // justifies that.
    if (g_fenceEvent != nullptr)
        g_fence->SetEventOnCompletion(value, g_fenceEvent);
    const bool completed = g_fenceEvent != nullptr && WaitForSingleObject(g_fenceEvent, 200) == WAIT_OBJECT_0;

    if (!completed)
    {
        readback->Release();
        device->Release();
        g_sampleDead.store(true, std::memory_order_relaxed);
        LOG_WARN("NrBridge sample: wait did not complete -- sampling off, nothing concluded");
        return;
    }

    void* mapped = nullptr;
    const D3D12_RANGE range { 0, static_cast<SIZE_T>(total) };
    if (SUCCEEDED(readback->Map(0, &range, &mapped)) && mapped != nullptr)
    {
        const float scaleX = ReadFloat(params, NVSDK_NGX_Parameter_MV_Scale_X, 1.0f);
        const float scaleY = ReadFloat(params, NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);

        const UINT w = static_cast<UINT>(desc.Width), h = desc.Height;
        const UINT xs[3] = { w / 2, w / 4, (3 * w) / 4 };
        const UINT ys[3] = { h / 2, h / 4, (3 * h) / 4 };

        char line[512];
        size_t used = 0;
        for (int i = 0; i < 3; ++i)
        {
            const auto* px = reinterpret_cast<const unsigned short*>(
                static_cast<const char*>(mapped) + foot.Offset + static_cast<UINT64>(ys[i]) * foot.Footprint.RowPitch +
                static_cast<UINT64>(xs[i]) * 4);
            const float vx = HalfToFloat(px[0]);
            const float vy = HalfToFloat(px[1]);
            used += static_cast<size_t>(std::snprintf(
                line + used, sizeof(line) - used, " (%u,%u) raw=(%.4f,%.4f) px=(%.2f,%.2f)", xs[i], ys[i], vx, vy,
                vx * scaleX, vy * scaleY));
        }

        LOG_INFO("NrBridge sample {}x{} fmt={} scale=({},{}){}", w, h, static_cast<unsigned>(desc.Format), scaleX,
                 scaleY, line);

        readback->Unmap(0, nullptr);
    }

    readback->Release();
    device->Release();
}
} // namespace NrBridge
