#pragma once

// The neural pass, packaged as one object the between slot can ask for a frame.
//
// Three things live here that the bridge itself has no business knowing:
//
//   - the network, as `nr_net.dll`, the replay of the plan that was recorded offline
//     (notes/optiscaler-bridge.md 27). Its shape: host features in, host head out, one
//     call that blocks for the whole of the device's cost.
//   - the two host passes around it, as `libnr_image.dll`: the sixteen feature channels
//     built from the frame's colour and its history, and the temporal composition of the
//     head back into the frame. Both are C ABI and both are the same bytes as the Python
//     they were transcribed from.
//   - the pieces the Python had and the C does not: the per-extent noise, the gate table,
//     the mirror onto the network's extent, and the reprojection of the history.
//
// Everything is loaded when it is first asked for, never before, and a failure to load is
// a reason returned, not a crash: the bridge's answer to a pass that will not start is the
// frame it was given.

#include <windows.h>

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace NrBridge
{

// Shapes of `nr_net.dll`'s nine exports. A copy of src/gpu/nr_net.h in the
// dlss-nr-on-a770 tree rather than an include of it, because the two are separate
// repositories; a change to the shape there has to be made here as well. They are taken
// by name at load time, so a build that disagrees with this file simply fails to resolve
// rather than calling with the wrong arguments.
extern "C"
{
    struct nr_net;
    typedef struct nr_net nr_net;

    typedef nr_net* (*NrNetOpen)(const char* library, const char* folder, const char* blobs);
    typedef const char* (*NrNetError)(nr_net*);
    typedef void (*NrNetClose)(nr_net*);
    typedef void (*NrNetExtent)(nr_net*, unsigned* height, unsigned* width);
    typedef void* (*NrNetFeatures)(nr_net*);
    typedef unsigned long long (*NrNetFeaturesBytes)(nr_net*);
    typedef const float* (*NrNetHead)(nr_net*);
    typedef unsigned long long (*NrNetHeadBytes)(nr_net*);
    typedef int (*NrNetRun)(nr_net*);

    // The fused path's two: the game's buffers by handle before the open, and the game's
    // fence by handle before a run.
    typedef void (*NrNetFuse)(unsigned long long colour, unsigned long long motion,
                              unsigned long long result, unsigned long long colourBytes,
                              unsigned long long motionBytes, unsigned long long resultBytes,
                              unsigned pitch, unsigned sourceHeight, unsigned sourceWidth,
                              float exposure, float intensity, float confidence,
                              const void* plan, unsigned long long planBytes, const float* table);
    typedef int (*NrNetWait)(nr_net*, unsigned long long handle, unsigned long long value);
    typedef int (*NrNetExposure)(nr_net*, float, int fresh);
    typedef int (*NrNetProfile)(nr_net*, int on);
    typedef unsigned (*NrNetProfileEach)(nr_net*, unsigned* kinds, double* ms, unsigned cap);
    typedef void (*NrNetProfileReset)(nr_net*);
    typedef int (*NrNetTimeSubmit)(nr_net*, int on);
    typedef int (*NrNetTimeReport)(nr_net*, double* out, unsigned cap);
    typedef int (*NrNetProbe)(nr_net*, int which, float* out);
    typedef int (*NrNetProbeFeatures)(nr_net*, float* out);
    // Whether the recording really carries the two fused passes. nr_net_fuse is the
    // request and this is the answer: the walk decides while it runs, so a fuse that was
    // asked for is not necessarily a network that was fused.
    typedef int (*NrNetFused)(nr_net*);

    // The device, before there is a network. The host's own Vulkan side is built on
    // the network's device: one device and one queue for both halves, as in the
    // reference. Taken by name, so a library without them simply does not offer it.
    typedef nr_net* (*NrNetBoot)(const char* library, const char* folder);
    typedef unsigned long long (*NrNetHandle)(nr_net*);
    typedef unsigned (*NrNetFamily)(nr_net*);
}

// The device the network runs on, as plain numbers so this header owes nothing to
// the Vulkan side that will use them.
struct DeviceHandles
{
    unsigned long long instance = 0;
    unsigned long long physical = 0;
    unsigned long long device = 0;
    unsigned long long queue = 0;
    unsigned family = 0;
};

// The two passes of `libnr_image.dll` the frame needs. `ptrdiff_t` is the C side's
// stride type and is what its ABI actually puts on the stack.
typedef void (*NrFeaturesHalf)(const float* colour, ptrdiff_t sy, ptrdiff_t sx, ptrdiff_t sc,
                               const float* history, ptrdiff_t ty, ptrdiff_t tx, ptrdiff_t tc,
                               const int32_t* rows, const int32_t* columns, ptrdiff_t height,
                               ptrdiff_t width, const float* noise, const float* controls,
                               uint16_t* output);

typedef void (*NrComposeTemporal)(const float* head, ptrdiff_t hy, ptrdiff_t hx, ptrdiff_t hc,
                                  const float* colour, ptrdiff_t sy, ptrdiff_t sx, ptrdiff_t sc,
                                  const float* history, ptrdiff_t ry, ptrdiff_t rx, ptrdiff_t rc,
                                  const float* previous, ptrdiff_t py, ptrdiff_t px, ptrdiff_t pc,
                                  const float* gate, ptrdiff_t gy, ptrdiff_t gx, const float* table,
                                  float confidence, const float* mask, ptrdiff_t my, ptrdiff_t mx,
                                  ptrdiff_t height, ptrdiff_t width, float intensity, float scale,
                                  float hold, float slope, float release, const float* grade,
                                  float* neural, float* output);

class Pipeline
{
  public:
    ~Pipeline();

    // Load the two libraries, build the graph and take the frame's own buffers. Slow, and
    // once. `folder` is where the network's pieces are: libxmx, the .spv files and the
    // plan's blob file, all of which were written there by the dump.
    //
    // `width` and `height` are the frame's, and the network's extent is derived from them
    // -- it is the frame padded out to what the graph can halve, which is not a choice
    // here but a property of the plan that was recorded (notes 28.1).
    bool Prepare(UINT width, UINT height, const std::filesystem::path& folder, std::string& why);
    bool Ready() const { return ready_; }

    // The network's per-pass GPU timing, which libxmx already keeps. One frame in three
    // costs 108ms against 62ms for the others and the totals cannot say why; this can.
    void StartProfiling();
    // The submit's three costs, and their report. These replace the pass timing: a frame
    // that is slow in `vkQueueSubmit` has every pass looking quick, which is what the
    // totals showed, and the pass timing itself is not free -- 417 timestamp writes and a
    // waiting read-back per frame, which the player feels as input lag.
    void StartSubmitTiming();
    void LogSubmitTiming();
    // The slowest passes since the last reset, sorted into `log`. `top` of them.
    void LogSlowestPasses(int top);

    // Bring the device up on its own and hand it over, so the hand-over can be built
    // on it instead of beside it. False where the network is off or the library is an
    // older one without the call; the caller then builds a device of its own, which
    // is the shape that reads the wrong bytes, but it is better than no device.
    bool Boot(const std::filesystem::path& folder, DeviceHandles* out);
    void Release();

    UINT Width() const { return width_; }
    UINT Height() const { return height_; }
    UINT NetWidth() const { return netW_; }
    UINT NetHeight() const { return netH_; }
    size_t ColourBytes() const { return size_t(width_) * height_ * 3 * sizeof(float); }

    // One frame, in place for the caller's buffers.
    //
    // `colour` is the frame, RGB float32, width*height*3, in [0,1] as the network expects
    // it. `motion` is the per-pixel shift to the same pixel in the previous frame, in
    // pixels, width*height*2, or null where the frame has none -- and with it null, or
    // `reproject` false, the history stands where it is, which is the identity the layer
    // has always used and the thing this host exists to replace. `output` is where the
    // composed frame goes, the same shape as `colour`.
    bool Run(const float* colour, const float* motion, float* output, bool reproject, std::string& why);

    // The frame's own buffers, imported by handle, so that the network reads the colour
    // where the game put it and writes the answer where the upscaler will read it.
    //
    // This is Mochi's Kind::Pre and Kind::Post: two passes at either end of the plan,
    // which is what lets a frame reach the network without the host ever seeing a pixel.
    // Call before the network is opened, because it decides what the recording is made of
    // — after it, Run is not the way to run a frame, RunFused is. The libraries are
    // loaded here when they are not already, which is what makes calling it first legal:
    // `nr_net_fuse` lives in the same library the open does.
    //
    // `pitch` is the colour's row in 32-bit words, which is how R11G11B10 is walked.
    bool Fuse(const std::filesystem::path& folder, unsigned long long colour,
              unsigned long long motion, unsigned long long result,
              unsigned long long colourBytes, unsigned long long motionBytes,
              unsigned long long resultBytes, UINT width, UINT height, UINT pitch, float exposure,
              std::string& why);
    // Asked of the network rather than remembered from the request. A host that believed
    // its own request while the network had never fused ran the fused path against a
    // recording with no Pre and no Post in it, with the staging the old path needed
    // already skipped: that hung the slot for a minute on 2026-10-13.
    bool Fused() const;

    // This frame's pre-exposure, for the two fused passes to divide and multiply by.
    // The game moves it between frames, so it cannot be given once at the open.
    void Exposure(float value, bool fresh);

    // What is in the answer: mean, lowest, highest. A diagnostic, and the only one that
    // can tell a network which is running and answering from one which is running and
    // answering with noise, since a fused frame never comes back to the host.
    bool Probe(int which, float* mean, float* low, float* high);
    // The network's own input, 16 channels x (mean, low, high) into 48 floats. Diagnostic.
    void ProbeFeatures(float* channels);

    // One frame on the fused path: the graph waits for the game's fence on the device,
    // runs, and writes the answer straight into the game's own buffer. Nothing is copied
    // in and nothing is read back, so there is no stage to time but the graph's.
    bool RunFused(unsigned long long fence, unsigned long long value, std::string& why);

    // What the frame cost, split the way the daemon's own stage log splits it.
    double FeaturesMs() const { return featuresMs_; }
    double NetMs() const { return netMs_; }
    double ComposeMs() const { return composeMs_; }
    // A look at what the network answered, for the log: the mean of the head's first three
    // channels and the largest absolute value in them. A network that has stopped answering
    // reads zero, and one that has started answering NaN reads 65504 -- both are visible
    // here and neither is visible in the frame.
    void HeadStats(float& mean, float& peak) const;
    // The share of the frame the gate held, as the daemon prints it: the mean of the
    // history weight over every fourth row. 0 on a frame with no history.
    float GateMean() const { return gateMean_; }

  private:
    bool LoadLibraries(const std::filesystem::path& folder, std::string& why);
    void BuildTables();
    void ResetHistory(const float* colour);

    bool ready_ = false;
    std::string why_;

    UINT width_ = 0, height_ = 0;
    UINT netW_ = 0, netH_ = 0;

    // The two libraries, held for the process: they are not unloaded while a frame could
    // still be running on the device they built.
    HMODULE net_ = nullptr;
    HMODULE image_ = nullptr;

    nr_net* net = nullptr;
    nr_net* booted_ = nullptr;    // the device, before any network was built on it
    NrNetOpen netOpen = nullptr;
    NrNetError netError = nullptr;
    NrNetClose netClose = nullptr;
    NrNetExtent netExtent = nullptr;
    NrNetFeatures netFeatures = nullptr;
    NrNetFeaturesBytes netFeaturesBytes = nullptr;
    NrNetHead netHead = nullptr;
    NrNetHeadBytes netHeadBytes = nullptr;
    NrNetRun netRun = nullptr;
    NrNetFuse netFuse = nullptr;
    NrNetWait netWait = nullptr;
    NrNetFused netFused = nullptr;
    NrNetExposure netExposure = nullptr;
    NrNetProfile netProfile = nullptr;
    NrNetProfileEach netProfileEach = nullptr;
    NrNetProfileReset netProfileReset = nullptr;
    NrNetTimeSubmit netTimeSubmit = nullptr;
    NrNetTimeReport netTimeReport = nullptr;
    NrNetProbe netProbe = nullptr;
    NrNetProbeFeatures netProbeFeatures = nullptr;
    NrNetBoot netBoot = nullptr;
    NrNetHandle netInstance = nullptr, netPhysical = nullptr, netDevice = nullptr, netQueue = nullptr;
    NrNetFamily netFamily = nullptr;

    NrFeaturesHalf features = nullptr;
    NrComposeTemporal compose = nullptr;

    // The frame's own state, all of it host memory.
    std::vector<int32_t> rows_, columns_; // the mirror onto the network's extent
    std::vector<float> noise_;            // (netH, netW, 3), from the extent and nothing else
    std::vector<float> gateTable_;        // the gate on every one of the 65536 half logits
    std::vector<float> history_;          // the last frame's prediction, at the frame's extent
    std::vector<float> previous_;         // the last frame's colour, for the hold and release
    std::vector<float> warped_;           // the history, moved by the frame's motion
    std::vector<float> neural_;           // this frame's prediction, kept for the next one
    bool haveHistory_ = false;

    // The fused path: the plan the input pass reads, laid out as that shader reads it —
    // the noise, then the controls, then the mirror's rows and its columns.
    bool fused_ = false;
    std::vector<float> plan_;

    double featuresMs_ = 0, netMs_ = 0, composeMs_ = 0;
    float gateMean_ = 0;
};

// The blend scale the recovered package uses, halved as the gate is (`notes/phase12`).
// A value already representable as a half.
inline float BlendScale() { return 0.73974609375f; }

// A half, either way: the sixteen bits the device, the feature buffer and the vector
// texture all agree on, and the float an arithmetic wants.
float HalfValue(uint16_t half);
uint16_t HalfOf(float value);

// The frame's colour, in and out of the format this game hands it over in.
//
// `DecodeColour` takes the rows the copy put the texture's top-left into -- `pitch` bytes
// to a row, which is not the width's -- and writes (width * height * 3) float32 RGB into
// `out`, undoing the game's pre-exposure and clipping what is left to [0,1]. `lowest` and
// `highest` are where it says what it had to clip, or null.
//
// `EncodeColour` is the way back: the composed frame, pre-exposure put back, written as
// the same packed rows.
void DecodeColour(const uint8_t* rows, UINT width, UINT height, size_t pitch, float preExposure, float* out,
                  float* lowest, float* highest);
void EncodeColour(const float* in, UINT width, UINT height, size_t pitch, float preExposure, uint8_t* rows);

} // namespace NrBridge
