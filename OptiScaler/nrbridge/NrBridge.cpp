#include "pch.h"

#include <nrbridge/NrBridge.h>

#include <atomic>
#include <cstdio>
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

std::atomic<unsigned long long> g_seen { 0 };
std::atomic<bool> g_contractLogged { false };
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
} // namespace NrBridge
