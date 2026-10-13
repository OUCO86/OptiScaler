#include <utility>
#include <vector>
#include "pch.h"

#include <nrbridge/NrPipeline.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

// The pass, at the bottom of the between slot: the network, the two host passes either
// side of it, and the handful of things the Python had that are not in either library.
//
// Everything numeric here is a transcription, not a design. Where the reference computed
// a value in NumPy this computes the same value in C, in the same order, and where the
// reference rounded to a half this rounds to a half in the same way -- round to nearest
// even, the way `numpy.float16` does and the way the GPU's own to_half pass does.
namespace
{

using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point since)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - since).count();
}

// ---- half --------------------------------------------------------------------------
//
// A half is carried as its sixteen bits, which is the one representation both sides agree
// on: it is what `nr_features_half` writes, what the gate table is indexed by, and what
// the device's own buffers hold.

uint32_t BitsOf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    return bits;
}

float FloatOf(uint32_t bits)
{
    float value = 0;
    std::memcpy(&value, &bits, sizeof bits);
    return value;
}

// Round to nearest even, ties to even -- the rule both NumPy's cast and the GPU's
// conversion follow, so the two agree bit for bit and this can stand beside either.
uint16_t HalfOf(float value)
{
    const uint32_t bits = BitsOf(value);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t biased = int32_t((bits >> 23) & 0xFFu);
    uint32_t mantissa = bits & 0x7FFFFFu;

    if (biased == 0xFF)
        return uint16_t(sign | 0x7C00u | (mantissa != 0 ? 0x200u : 0u)); // NaN keeps a payload

    const int32_t exponent = biased - 127 + 15;

    if (exponent >= 31) // too large for a half; infinity, as the cast gives
        return uint16_t(sign | 0x7C00u);

    if (exponent <= 0)
    {
        // Subnormal, or a value that rounds up to the smallest normal. The implicit bit
        // is put back and the shift grows by one bit for every step below it.
        if (exponent < -10)
            return uint16_t(sign);

        mantissa |= 0x800000u;
        const uint32_t shift = uint32_t(14 - exponent);
        uint32_t half = mantissa >> shift;
        const uint32_t rest = mantissa & ((1u << shift) - 1u);
        const uint32_t halfWay = 1u << (shift - 1);
        if (rest > halfWay || (rest == halfWay && (half & 1u) != 0))
            half += 1;
        return uint16_t(sign | half);
    }

    uint32_t half = (uint32_t(exponent) << 10) | (mantissa >> 13);
    const uint32_t rest = mantissa & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (half & 1u) != 0))
        half += 1; // may carry into the exponent, which is the right answer

    return uint16_t(sign | (half & 0xFFFFu));
}

float HalfValue(uint16_t half)
{
    const uint32_t sign = uint32_t(half & 0x8000u);
    const uint32_t biased = (half >> 10) & 0x1Fu;
    const uint32_t mantissa = half & 0x3FFu;

    if (biased == 0)
    {
        // Zero, or subnormal: mantissa * 2^-24, which is the half's own rule.
        const float magnitude = float(mantissa) * (1.0f / 16777216.0f);
        return sign != 0 ? -magnitude : magnitude;
    }

    if (biased == 31)
    {
        // Infinity and NaN both reach the gate table. An infinite logit gates fully; a
        // NaN one is never a real answer from the network, and a zero in its place keeps
        // it from spreading through the frame.
        const float magnitude = mantissa != 0 ? 0.0f : 1.0e30f;
        return sign != 0 ? -magnitude : magnitude;
    }

    return FloatOf((sign << 16) | ((biased - 15 + 127) << 23) | (mantissa << 13));
}

// ---- the pieces the Python had -----------------------------------------------------

// `_dynamic_shift_mix` and `_uniform24`, from work/mlx-dlss/.../features.py. A hash of
// the pixel's place, not a random number: the same extent always gives the same noise,
// which is why it can be built once.
uint32_t ShiftMix(uint32_t value)
{
    const uint32_t shift = (value >> 28) + 4u;
    const uint32_t mixed = value ^ (value >> shift);
    return mixed * 0x108EF2D9u;
}

float Uniform24(uint32_t value)
{
    const uint32_t mixed = ShiftMix(value);
    const uint32_t bits = (mixed >> 30) ^ (mixed >> 8);
    return float(bits + 1u) * 5.960464477539063e-8f; // 2^-24
}

// The three channels, each rounded to a half, as `deterministic_noise` stacks them.
void NoiseChannel(uint32_t seed, float* out)
{
    const uint32_t mixed = ShiftMix(seed) ^ (ShiftMix(seed) >> 22);

    const float radiusA = std::sqrt(-2.0f * std::log(Uniform24(mixed * 0xCAA5B80Du + 0x21DD796Bu)));
    const float radiusB = std::sqrt(-2.0f * std::log(Uniform24(mixed * 0x2C9277B5u + 0xAC564B05u)));
    const float angleA = 6.2831854820251465f * Uniform24(mixed * 0xFA6DC5F9u + 0x4712A88Eu);
    const float angleB = 6.2831854820251465f * Uniform24(mixed * 0x83232C31u + 0x3463E0ACu);

    out[0] = HalfValue(HalfOf(radiusB * std::cos(angleA)));
    out[1] = HalfValue(HalfOf(radiusB * std::sin(angleA)));
    out[2] = HalfValue(HalfOf(radiusA * std::cos(angleB)));
}

// The mirror onto the network's extent, `NetworkGeometry.extended_indices`: a pixel past
// the frame's edge takes the one the same distance inside it, which is what the vendor
// pads a field with.
int32_t MirrorIndex(int32_t index, int32_t extent)
{
    return index < extent ? index : (std::max)(0, 2 * extent - 2 - index);
}

// ---- the colour's format -----------------------------------------------------------
//
// R11G11B10_FLOAT is three unsigned floats sharing thirty-two bits: five bits of exponent
// biased by fifteen, no sign, and six, six and five bits of mantissa.

float FromUnsignedFloat(uint32_t packed, uint32_t mantissaBits)
{
    const uint32_t exponent = packed >> mantissaBits;
    const uint32_t mantissa = packed & ((1u << mantissaBits) - 1u);

    if (exponent == 0)
        return float(mantissa) * std::exp2(float(1 - 15 - int32_t(mantissaBits)));

    if (exponent == 31)
        return mantissa == 0 ? 1.0e30f : 0.0f; // infinity, and a NaN read as nothing

    return float(1.0 + double(mantissa) / double(1u << mantissaBits)) * std::exp2(float(exponent) - 15.0f);
}

uint32_t ToUnsignedFloat(float value, uint32_t mantissaBits)
{
    if (!(value > 0.0f)) // zero, negative and NaN all encode as zero: there is no sign
        return 0;

    const int32_t exponent = int32_t(std::floor(std::log2(value)));
    const int32_t biased = exponent + 15;

    if (biased >= 31)
        return 31u << mantissaBits; // clamp to the largest the format holds

    uint32_t mantissa;
    if (biased <= 0)
    {
        // Subnormal: the value the format holds below its smallest exponent, which is
        // mantissa * 2^(1 - 15 - mantissaBits).
        mantissa = uint32_t(value * std::exp2(float(mantissaBits + 14)) + 0.5f);
        if (mantissa >= (1u << mantissaBits))
            mantissa = (1u << mantissaBits) - 1u;
        return mantissa;
    }

    const double scaled = std::ldexp(double(value), -exponent) - 1.0;
    mantissa = uint32_t(scaled * double(1u << mantissaBits) + 0.5);
    if (mantissa >= (1u << mantissaBits))
    {
        mantissa = 0;
        return uint32_t(biased + 1) << mantissaBits; // the carry into the exponent
    }

    return (uint32_t(biased) << mantissaBits) | mantissa;
}

} // namespace

namespace NrBridge
{

// The two spellings the rest of the host may want, out of the anonymous namespace above.
float HalfValue(uint16_t half) { return ::HalfValue(half); }
uint16_t HalfOf(float value) { return ::HalfOf(value); }

// The colour the game hands over is the scene's, scaled by the game's pre-exposure and
// left in a format with no sign and eleven bits to a channel. Undoing the scale is what
// FSR does with the same input (`PrepareRgb`: `fRgb /= fPreExposure`, in the host's own
// FSR3 shader headers), and it is what the network needs: it was not trained on a frame
// that had been dimmed by however much this scene was dimmed by.
//
// What is left after that can still be above one, where the scene is brighter than the
// display. There is no tone curve here -- the vendor's own preprocess, which this tree
// does not have, is what puts one in -- so the frame is clipped, and the log is told how
// much of it had to be.
void DecodeColour(const uint8_t* rows, UINT width, UINT height, size_t pitch, float preExposure,
                  float* out, float* lowest, float* highest)
{
    const float scale = preExposure > 0.0f ? 1.0f / preExposure : 1.0f;
    float low = 0.0f, high = 0.0f;

    for (UINT y = 0; y < height; ++y)
    {
        const uint32_t* line = reinterpret_cast<const uint32_t*>(rows + y * pitch);
        float* target = out + size_t(y) * width * 3;

        for (UINT x = 0; x < width; ++x)
        {
            const uint32_t packed = line[x];
            float value[3] = { FromUnsignedFloat(packed & 0x7FFu, 6) * scale,
                               FromUnsignedFloat((packed >> 11) & 0x7FFu, 6) * scale,
                               FromUnsignedFloat((packed >> 22) & 0x3FFu, 5) * scale };

            for (int c = 0; c < 3; ++c)
            {
                if (value[c] < 0.0f)
                    value[c] = 0.0f;
                else if (value[c] > 1.0f)
                    value[c] = 1.0f;

                if (value[c] < low)
                    low = value[c];
                if (value[c] > high)
                    high = value[c];

                target[x * 3 + c] = value[c];
            }
        }
    }

    if (lowest != nullptr)
        *lowest = low;
    if (highest != nullptr)
        *highest = high;
}

void EncodeColour(const float* in, UINT width, UINT height, size_t pitch, float preExposure, uint8_t* rows)
{
    for (UINT y = 0; y < height; ++y)
    {
        uint32_t* line = reinterpret_cast<uint32_t*>(rows + y * pitch);
        const float* source = in + size_t(y) * width * 3;

        for (UINT x = 0; x < width; ++x)
        {
            const float r = source[x * 3 + 0] * preExposure;
            const float g = source[x * 3 + 1] * preExposure;
            const float b = source[x * 3 + 2] * preExposure;

            line[x] = ToUnsignedFloat(r, 6) | (ToUnsignedFloat(g, 6) << 11) | (ToUnsignedFloat(b, 5) << 22);
        }
    }
}

Pipeline::~Pipeline()
{
    Release();
}

// Both libraries are loaded from the folder the network's pieces were written to, and
// both carry a runtime beside them -- libxmx brings libgomp and libwinpthread, and
// libnr_image needs the same OpenMP runtime. A LoadLibrary from anywhere else finds the
// DLL and not what it needs.
//
// The directory the process was using is put back afterwards. `nr_net_open` sets it too,
// and a host that changed the search path behind the game's back would change what every
// later LoadLibrary in the process resolves to.
bool Pipeline::LoadLibraries(const std::filesystem::path& folder, std::string& why)
{
    char previous[MAX_PATH] {};
    DWORD length = GetDllDirectoryA(sizeof previous, previous);
    if (length == 0 || length > sizeof previous)
        previous[0] = '\0';

    const std::string directory = folder.string();
    SetDllDirectoryA(directory.c_str());

    const std::string imagePath = (folder / "libnr_image.dll").string();
    const std::string netPath = (folder / "nr_net.dll").string();

    image_ = LoadLibraryA(imagePath.c_str());
    net_ = image_ != nullptr ? LoadLibraryA(netPath.c_str()) : nullptr;

    SetDllDirectoryA(previous);

    if (image_ == nullptr)
    {
        why = "libnr_image.dll would not load (" + imagePath + ")";
        return false;
    }
    if (net_ == nullptr)
    {
        why = "nr_net.dll would not load (" + netPath + ")";
        return false;
    }

    features = reinterpret_cast<NrFeaturesHalf>(GetProcAddress(image_, "nr_features_half"));
    compose = reinterpret_cast<NrComposeTemporal>(GetProcAddress(image_, "nr_compose_temporal"));

    netOpen = reinterpret_cast<NrNetOpen>(GetProcAddress(net_, "nr_net_open"));
    netError = reinterpret_cast<NrNetError>(GetProcAddress(net_, "nr_net_error"));
    netClose = reinterpret_cast<NrNetClose>(GetProcAddress(net_, "nr_net_close"));
    netExtent = reinterpret_cast<NrNetExtent>(GetProcAddress(net_, "nr_net_extent"));
    netFeatures = reinterpret_cast<NrNetFeatures>(GetProcAddress(net_, "nr_net_features"));
    netFeaturesBytes = reinterpret_cast<NrNetFeaturesBytes>(GetProcAddress(net_, "nr_net_features_bytes"));
    netHead = reinterpret_cast<NrNetHead>(GetProcAddress(net_, "nr_net_head"));
    netHeadBytes = reinterpret_cast<NrNetHeadBytes>(GetProcAddress(net_, "nr_net_head_bytes"));
    netRun = reinterpret_cast<NrNetRun>(GetProcAddress(net_, "nr_net_run"));
    netFuse = reinterpret_cast<NrNetFuse>(GetProcAddress(net_, "nr_net_fuse"));
    netWait = reinterpret_cast<NrNetWait>(GetProcAddress(net_, "nr_net_wait"));
    netFused = reinterpret_cast<NrNetFused>(GetProcAddress(net_, "nr_net_fused"));
    netExposure = reinterpret_cast<NrNetExposure>(GetProcAddress(net_, "nr_net_exposure"));
    netProfile = reinterpret_cast<NrNetProfile>(GetProcAddress(net_, "nr_net_profile"));
    netProfileEach = reinterpret_cast<NrNetProfileEach>(GetProcAddress(net_, "nr_net_profile_each"));
    netProfileReset = reinterpret_cast<NrNetProfileReset>(GetProcAddress(net_, "nr_net_profile_reset"));
    netTimeSubmit = reinterpret_cast<NrNetTimeSubmit>(GetProcAddress(net_, "nr_net_time_submit"));
    netTimeReport = reinterpret_cast<NrNetTimeReport>(GetProcAddress(net_, "nr_net_time_report"));
    netProbe = reinterpret_cast<NrNetProbe>(GetProcAddress(net_, "nr_net_probe"));
    netProbeFeatures = reinterpret_cast<NrNetProbeFeatures>(GetProcAddress(net_, "nr_net_probe_features"));
    netBoot = reinterpret_cast<NrNetBoot>(GetProcAddress(net_, "nr_net_boot"));
    netInstance = reinterpret_cast<NrNetHandle>(GetProcAddress(net_, "nr_net_instance"));
    netPhysical = reinterpret_cast<NrNetHandle>(GetProcAddress(net_, "nr_net_physical"));
    netDevice = reinterpret_cast<NrNetHandle>(GetProcAddress(net_, "nr_net_device"));
    netQueue = reinterpret_cast<NrNetHandle>(GetProcAddress(net_, "nr_net_queue"));
    netFamily = reinterpret_cast<NrNetFamily>(GetProcAddress(net_, "nr_net_family"));

    if (features == nullptr || compose == nullptr)
    {
        why = "libnr_image.dll does not have the two passes";
        return false;
    }
    if (netOpen == nullptr || netError == nullptr || netClose == nullptr || netExtent == nullptr ||
        netFeatures == nullptr || netFeaturesBytes == nullptr || netHead == nullptr || netHeadBytes == nullptr ||
        netRun == nullptr)
    {
        why = "nr_net.dll does not have the nine calls";
        return false;
    }

    return true;
}

bool Pipeline::Boot(const std::filesystem::path& folder, DeviceHandles* out)
{
    if (out == nullptr)
        return false;

    if (net_ == nullptr)
    {
        std::string loadWhy;
        if (!LoadLibraries(folder, loadWhy))
        {
            why_ = loadWhy;
            return false;
        }
    }
    if (netBoot == nullptr || netDevice == nullptr || netQueue == nullptr)
        return false;

    const std::string directory = folder.string();
    nr_net* up = netBoot("libxmx.dll", directory.c_str());
    if (up == nullptr)
        return false;
    if (netError != nullptr)
    {
        const char* bad = netError(up);
        if (bad != nullptr && *bad != '\0')
        {
            why_ = std::string("the network's device would not come up: ") + bad;
            return false;
        }
    }

    out->instance = netInstance != nullptr ? netInstance(up) : 0ull;
    out->physical = netPhysical != nullptr ? netPhysical(up) : 0ull;
    out->device = netDevice(up);
    out->queue = netQueue(up);
    out->family = netFamily != nullptr ? netFamily(up) : 0u;

    booted_ = up;
    return out->device != 0ull && out->queue != 0ull;
}

void Pipeline::ProbeFeatures(float* channels)
{
    if (netProbeFeatures == nullptr || net == nullptr || channels == nullptr)
        return;
    netProbeFeatures(net, channels);
}

bool Pipeline::Probe(int which, float* mean, float* low, float* high)
{
    if (netProbe == nullptr || net == nullptr || mean == nullptr || low == nullptr || high == nullptr)
        return false;
    float stats[3] = { 0.0f, 0.0f, 0.0f };
    if (netProbe(net, which, stats) != 0)
        return false;
    *mean = stats[0];
    *low = stats[1];
    *high = stats[2];
    return true;
}

void Pipeline::StartProfiling()
{
    LOG_INFO("NrBridge: asking the network to time its passes (have the call: {})",
             netProfile != nullptr);
    if (netProfile == nullptr || net == nullptr)
        return;
    // Asked once, at the first frame, and a refusal is not fatal: the network runs the same
    // either way, it just cannot say which of its passes was slow.
    netProfile(net, 1);
}

void Pipeline::StartSubmitTiming()
{
    if (netTimeSubmit == nullptr || net == nullptr)
        return;
    /* The pass timing is turned off as this is turned on, and not because the two cannot
     * both run: it costs 417 timestamp writes and a `vkGetQueryPoolResults` with WAIT per
     * frame, and that read-back blocks the CPU until the device is past the last query --
     * which the player sees as input lag. The submit's own cost is three clock reads. */
    if (netProfile != nullptr && net != nullptr)
        netProfile(net, 0);
    netTimeSubmit(net, 1);
}

void Pipeline::LogSubmitTiming()
{
    if (netTimeReport == nullptr || net == nullptr)
        return;
    double v[8] = {};
    const int frames = netTimeReport(net, v, 8);
    if (frames <= 0)
        return;
    const double n = static_cast<double>(frames);
    LOG_INFO("NrBridge submit: {} frames | submit {:.2f}ms total, {:.2f}ms max | "
             "wait {:.2f}ms total, {:.2f}ms max | collect {:.2f}ms total, {:.2f}ms max | "
             "mean {:.2f}ms",
             frames, v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
}

void Pipeline::LogSlowestPasses(int top)
{
    if (netProfileEach == nullptr || netProfileReset == nullptr || net == nullptr || top <= 0)
        return;

    // Every pass since the last look, 417 of them in a normal frame and more when several
    // have gone by. Kept on the stack in one allocation: the sizes are known here and the
    // network's own limit is far higher.
    enum { Cap = 4096 };
    std::vector<std::pair<double, unsigned>> passes;
    passes.reserve(Cap);
    unsigned kinds[Cap] {};
    double ms[Cap] {};
    const unsigned got = netProfileEach(net, kinds, ms, Cap);
    for (unsigned i = 0; i < got; i++)
        passes.emplace_back(ms[i], (kinds[i] << 16) | i);

    std::sort(passes.rbegin(), passes.rend());
    double total = 0.0;
    for (const auto& p : passes) total += p.first;

    LOG_INFO("NrBridge passes: {} timed, {:.2f}ms in total | slowest:", got, total);
    const unsigned take = static_cast<unsigned>(top) < passes.size() ? static_cast<unsigned>(top)
                                                                     : passes.size();
    for (unsigned i = 0; i < take; i++) {
        const unsigned kind = passes[i].second >> 16;
        const unsigned at = passes[i].second & 0xFFFFu;
        LOG_INFO("NrBridge pass #{}: family {} took {:.3f}ms", at, kind, passes[i].first);
    }

    netProfileReset(net);
}

void Pipeline::Exposure(float value, bool fresh)
{
    if (netExposure == nullptr || net == nullptr)
        return;
    if (!(value > 0.0f) || !std::isfinite(value))
        return;   // the passes then read the one the fused open put there
    netExposure(net, value, fresh ? 1 : 0);
}

bool Pipeline::Prepare(UINT width, UINT height, const std::filesystem::path& folder, std::string& why)
{
    if (ready_)
        return true;

    if (width == 0 || height == 0)
    {
        why = "the frame has no extent";
        return false;
    }

    if (!LoadLibraries(folder, why))
        return false;

    // The whole of the build: the device, the resident shaders, every buffer, the weights
    // read back into them, and the graph captured. A quarter second, once.
    //
    // The open sets the process's DLL search directory, so that libxmx finds the OpenMP
    // runtime beside it, and does not put it back. A host that left it changed would have
    // changed what every later LoadLibrary in the game resolves to, so it is read first
    // and restored straight after.
    char search[MAX_PATH] {};
    const DWORD length = GetDllDirectoryA(sizeof search, search);
    if (length == 0 || length > sizeof search)
        search[0] = '\0';

    const std::string library = (folder / "libxmx.dll").string();
    net = netOpen(library.c_str(), folder.string().c_str(), nullptr);

    SetDllDirectoryA(search);

    if (net == nullptr || netError(net) != nullptr)
    {
        why = net != nullptr ? std::string("the network would not start: ") + netError(net)
                             : "the network would not start";
        return false;
    }

    // The extent is the plan's, not the frame's: the graph was recorded at one size and
    // cannot be asked for another. The frame has to fit it, and the difference between
    // the two is the padding the mirror below fills.
    unsigned recordedHeight = 0, recordedWidth = 0;
    netExtent(net, &recordedHeight, &recordedWidth);
    if (recordedWidth != width || recordedHeight < height || recordedHeight - height >= 64)
    {
        char text[192];
        std::snprintf(text, sizeof text, "the frame is %ux%u and the network was recorded at %ux%u", width,
                      height, recordedWidth, recordedHeight);
        why = text;
        return false;
    }

    width_ = width;
    height_ = height;
    netW_ = recordedWidth;
    netH_ = recordedHeight;

    BuildTables();

    const size_t pixels = size_t(width_) * height_;
    history_.assign(pixels * 3, 0.0f);
    previous_.assign(pixels * 3, 0.0f);
    warped_.assign(pixels * 3, 0.0f);
    neural_.assign(pixels * 3, 0.0f);
    haveHistory_ = false;

    why_.clear();
    ready_ = true;
    return true;
}

void Pipeline::BuildTables()
{
    const size_t network = size_t(netH_) * netW_;

    rows_.resize(netH_);
    for (UINT y = 0; y < netH_; ++y)
        rows_[y] = MirrorIndex(int32_t(y), int32_t(height_));

    columns_.resize(netW_);
    for (UINT x = 0; x < netW_; ++x)
        columns_[x] = MirrorIndex(int32_t(x), int32_t(width_));

    // The noise depends on the extent and the frame index and on nothing else, and the
    // index does not move in a live frame, so it is built once (nr_frame._cached_noise).
    noise_.resize(network * 3);
    for (UINT y = 0; y < netH_; ++y)
    {
        const uint32_t row = uint32_t(y) * 0xD8163841u;
        for (UINT x = 0; x < netW_; ++x)
        {
            const uint32_t seed = row ^ (uint32_t(x) * 0x8DA6B343u) ^ 0x243F6A88u;
            NoiseChannel(seed, noise_.data() + (size_t(y) * netW_ + x) * 3);
        }
    }

    // The gate on every half logit there is. NumPy evaluates the sigmoid once on all
    // 65536 of them and indexes the answer per pixel, which is bit-identical to running
    // it per pixel because the logit is rounded to a half first (nr_frame.gate_table).
    // Building it with a double exponential and rounding at the end: expf and NumPy's
    // float32 exponential differ in the last bit, and this is the one place that
    // difference would reach every pixel of the frame.
    gateTable_.resize(1u << 16);
    const float scale = HalfValue(HalfOf(BlendScale()));
    for (uint32_t i = 0; i < (1u << 16); ++i)
    {
        const double logit = double(HalfValue(uint16_t(i)));
        const double gate = 1.0 / (1.0 + std::exp(-logit)) * double(scale);
        gateTable_[i] = float(std::clamp(gate, 0.0, 1.0));
    }
}

void Pipeline::ResetHistory(const float* colour)
{
    const size_t pixels = size_t(width_) * height_;
    std::memcpy(previous_.data(), colour, pixels * 3 * sizeof(float));
    haveHistory_ = false;
}

// The fused path: the frame's own buffers go into the recording, so that the network
// reads the colour where the game put it and writes the answer where the upscaler will
// read it. Nothing this puts together is a frame — it is state, and it goes in once.
bool Pipeline::Fuse(const std::filesystem::path& folder, unsigned long long colour,
                    unsigned long long motion, unsigned long long result,
                    unsigned long long colourBytes, unsigned long long motionBytes,
                    unsigned long long resultBytes, UINT width, UINT height, UINT pitch,
                    float exposure, std::string& why)
{
    // `nr_net_fuse` is an export of the library the open loads, so asking before the open
    // means loading first. This is the whole of the ordering there is to get right here:
    // the call has to land before `nr_net_open`, and the open cannot be moved ahead of it
    // because the open is what walks the plan the two passes are put into.
    if (net_ == nullptr)
    {
        std::string loadWhy;
        if (!LoadLibraries(folder, loadWhy))
        {
            why = loadWhy;
            return false;
        }
    }

    if (netFuse == nullptr || netWait == nullptr)
    {
        why = "the network library has no fused passes";
        return false;
    }
    if (net != nullptr)
    {
        why = "the network is already open, and the fused passes have to be asked for before it is";
        return false;
    }

    width_ = width;
    height_ = height;
    // The graph was recorded at one extent and cannot be asked for another, so the
    // extent is the plan's and not the frame's: the frame has to fit it, and the
    // difference is the padding the mirror fills. This is the same rounding
    // EnsureNetwork checks the plan's own extent against when it comes back.
    netW_ = width;
    netH_ = ((height + 31u) / 32u) * 32u;

    BuildTables();

    // What nr_pre.comp reads, in its order: the noise, the five controls, the mirror's
    // rows and its columns. The indices are 32-bit and ride in the same floats.
    const size_t noise = size_t(netH_) * netW_ * 3;
    plan_.assign(noise + 5 + size_t(netH_) + netW_, 0.0f);
    std::memcpy(plan_.data(), noise_.data(), noise * sizeof(float));
    float* at = plan_.data() + noise;
    const float controls[5] = { 0.0f, 1.0f, 1.0f, -1.0f, -1.0f };
    std::memcpy(at, controls, sizeof controls);
    at += 5;
    std::memcpy(at, rows_.data(), size_t(netH_) * sizeof(int32_t));
    at += netH_;
    std::memcpy(at, columns_.data(), size_t(netW_) * sizeof(int32_t));

    // width, height, then the row in 32-bit words: the order the signature has them. They
    // were given as pitch, height, width, which the network reads as a width of the row and
    // a row of the width -- the same three numbers in the wrong places, which this frame
    // has not shown because its row happens to be its width.
    netFuse(colour, motion, result, colourBytes, motionBytes, resultBytes, width, height, pitch,
            exposure, 1.0f, 1.0f, plan_.data(), plan_.size() * sizeof(float), gateTable_.data());
    fused_ = true;
    return true;
}

// Whether the recording really has the two passes in it. The request is not the answer:
// the walk puts them in while it runs, and a walk that could not leaves a network that
// has to be run the old way -- with the staging this path skips.
bool Pipeline::Fused() const
{
    if (!fused_)
        return false;
    if (netFused == nullptr)
        return net == nullptr; // nothing to ask yet: the open has not happened
    return net != nullptr && netFused(net) != 0;
}

// One frame on the fused path. The colour is the game's buffer, so the graph waits for
// the game's fence on the device — a semaphore wait inside the submit, not the host
// waiting and then submitting — and the answer is written where the upscaler will read
// it. There is no copy in, no read back, and no stage to time but the graph's.
bool Pipeline::RunFused(unsigned long long fence, unsigned long long value, std::string& why)
{
    if (net == nullptr || !fused_)
    {
        why = "the fused path is not up";
        return false;
    }

    const auto started = Clock::now();
    if (netWait(net, fence, value) != 0)
    {
        why = std::string("the graph would not wait for the frame: ") +
              (netError(net) != nullptr ? netError(net) : "");
        return false;
    }
    if (netRun(net) != 0)
    {
        why = std::string("the graph would not run: ") + (netError(net) != nullptr ? netError(net) : "");
        return false;
    }
    netMs_ = ElapsedMs(started);
    return true;
}

bool Pipeline::Run(const float* colour, const float* motion, float* output, bool reproject, std::string& why)
{
    if (!ready_)
    {
        why = why_.empty() ? "the pass is not up" : why_;
        return false;
    }

    const size_t pixels = size_t(width_) * height_;
    const ptrdiff_t sy = ptrdiff_t(width_) * 3;
    constexpr ptrdiff_t sx = 3, sc = 1;

    // Where the history is read from. With motion and reprojection on, it is moved first:
    // the pixel the frame's pixel was in the frame before, which is what a layer at
    // present has never been able to ask for and the whole reason this host exists.
    // Without either, it stands where it was -- the identity, which is right wherever the
    // scene stood still and wrong wherever it moved.
    const float* history = history_.data();
    if (motion != nullptr && reproject && haveHistory_)
    {
        for (size_t p = 0; p < pixels; ++p)
        {
            const float shiftX = motion[p * 2 + 0];
            const float shiftY = motion[p * 2 + 1];
            const int32_t x = int32_t(p % width_);
            const int32_t y = int32_t(p / width_);

            int32_t fromX = int32_t(std::floor(float(x) + shiftX + 0.5f));
            int32_t fromY = int32_t(std::floor(float(y) + shiftY + 0.5f));
            fromX = std::clamp(fromX, 0, int32_t(width_) - 1);
            fromY = std::clamp(fromY, 0, int32_t(height_) - 1);

            const float* source = history_.data() + (size_t(fromY) * width_ + fromX) * 3;
            std::memcpy(warped_.data() + p * 3, source, 3 * sizeof(float));
        }
        history = warped_.data();
    }
    else if (!haveHistory_)
    {
        // The first frame: channels 7-9 repeat the colour, which is exactly what the
        // vendor's own first-frame layout puts there (features.make_features).
        history = colour;
    }

    // The conditioning values: a style of 0 with tone and structure at 1, the `standard`
    // profile, and the vendor's constant -1 in the two channels this tree does not use.
    const float controls[5] = { 0.0f, 1.0f, 1.0f, -1.0f, -1.0f };

    auto started = Clock::now();
    uint16_t* const input = static_cast<uint16_t*>(netFeatures(net));
    features(colour, sy, sx, sc, history, sy, sx, sc, rows_.data(), columns_.data(), netH_, netW_,
             noise_.data(), controls, input);
    featuresMs_ = ElapsedMs(started);

    started = Clock::now();
    if (netRun(net) != 0)
    {
        why = netError(net) != nullptr ? std::string("the network would not run: ") + netError(net)
                                       : "the network would not run";
        return false;
    }
    netMs_ = ElapsedMs(started);

    const float* head = netHead(net);
    const ptrdiff_t hy = ptrdiff_t(netW_) * 4;

    // The composition, and the frame's own numbers for it: the daemon's defaults
    // (--intensity 1, --temporal 1, --hold 1, --release 24), and the two folded constants
    // that stand for them -- the hold's ramp of four levels of 255, and the release's fall
    // to nothing by twenty-four.
    started = Clock::now();
    compose(head, hy, 4, 1, colour, sy, sx, sc, history, sy, sx, sc, previous_.data(), sy, sx, sc, nullptr, 0, 0,
            gateTable_.data(), 1.0f, nullptr, 0, 0, height_, width_, 1.0f, BlendScale(), 1.0f,
            -255.0f * 1.0f / 4.0f, -255.0f / 24.0f, nullptr, neural_.data(), output);
    composeMs_ = ElapsedMs(started);

    // What the gate made of the frame, over every fourth row, the way the daemon meters
    // it: a frame whose history is right everywhere reads about 0.7, and one whose
    // history is wrong reads about 0.03.
    double sum = 0.0;
    size_t count = 0;
    for (UINT y = 0; y < height_; y += 4)
    {
        const float* line = head + size_t(y) * hy;
        for (UINT x = 0; x < width_; ++x)
        {
            const uint16_t logit = HalfOf(line[x * 4 + 3]);
            sum += double(gateTable_[logit]);
            ++count;
        }
    }
    gateMean_ = count != 0 ? float(sum / double(count)) : 0.0f;

    // This frame becomes the next one's past.
    std::memcpy(previous_.data(), colour, pixels * 3 * sizeof(float));
    std::memcpy(history_.data(), neural_.data(), pixels * 3 * sizeof(float));
    haveHistory_ = true;

    why.clear();
    return true;
}

void Pipeline::HeadStats(float& mean, float& peak) const
{
    mean = 0.0f;
    peak = 0.0f;
    if (!ready_)
        return;

    const float* head = netHead(net);
    const unsigned long long bytes = netHeadBytes(net);
    if (head == nullptr || bytes == 0)
        return;

    const size_t count = size_t(bytes / sizeof(float));
    double sum = 0.0;
    float largest = 0.0f;
    size_t used = 0;

    for (size_t i = 0; i < count; i += 4)
    {
        for (int c = 0; c < 3; ++c)
        {
            const float value = head[i + c];
            sum += double(value);
            const float magnitude = std::fabs(value);
            if (magnitude > largest)
                largest = magnitude;
            ++used;
        }
    }

    if (used != 0)
        mean = float(sum / double(used));
    peak = largest;
}

void Pipeline::Release()
{
    if (net != nullptr && netClose != nullptr)
        netClose(net);
    net = nullptr;

    if (net_ != nullptr)
        FreeLibrary(net_);
    if (image_ != nullptr)
        FreeLibrary(image_);
    net_ = image_ = nullptr;

    ready_ = false;
}

} // namespace NrBridge
