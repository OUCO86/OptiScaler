#pragma once

// Neural rendering over the upscaler's inputs, before the upscaler consumes them.
//
// This file is a transcription of the reference integration, not a design of ours. Every
// structure, every field and every method below is taken from
// refs/optiscaler/ta_AmdPreSr.h (Frame, Settings, LookSettings, RtgiSettings, Backend) and
// refs/optiscaler/ta_DlssNrFeature_Dx12.h (EvaluateBeforeUpscale, Shutdown, IsRunning,
// FailureReason, RetryAfterFailure, MaxPassCount). Field names and default values are kept
// as they are there, so that a diff between the two stays a diff of behaviour, not of shape.
//
// The one deliberate difference, and the only one allowed: the reference runs on an AMD card
// and its upscaler output is FSR. Here the output upscaler is XeSS, because that is what the
// A770 accelerates natively. Nothing here chooses an upscaler -- OptiScaler's own path does
// that, and this pass only ever sees its inputs.
//
// The pass runs pre-SR, exactly as the reference defaults (its [DlssNr] RunBeforeSR=true):
// it is handed the low-resolution colour before Super Resolution, with depth and motion
// vectors.

#include <d3d12.h>
#include <nvsdk_ngx.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>

class Config;

namespace NrBridge
{
inline constexpr unsigned int MaxPassCount = 3;

struct Frame
{
    ID3D12Resource *colour = nullptr, *motion = nullptr, *depth = nullptr, *exposure = nullptr;
    UINT width = 0, height = 0;
    float motionScaleX = 1, motionScaleY = 1;
    // Active display extent, excluding allocation padding; zero means render-resolution vectors.
    UINT motionWidth = 0, motionHeight = 0;
    float jitterX = 0, jitterY = 0;
    bool motionJittered = false, temporalInputsValid = false;
    float preExposure = 1, exposureScale = 1;
    bool reset = false, depthInverted = false;
    D3D12_RESOURCE_STATES colourState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    D3D12_RESOURCE_STATES motionState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    D3D12_RESOURCE_STATES depthState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    D3D12_RESOURCE_STATES exposureState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
};

struct LookSettings
{
    bool enabled = false;
    UINT appearance = 2, inspect = 0;
    bool detectSkin = true;
    float mix = 1, materialDetail = 1.15f, shapeDefinition = 1.2f, localLighting = 1.15f;
    float skinDetail = 1.1f, skinSoftness = .486f, specularControl = .58f, highlightRollOff = .9f;
    float colourSeparation = 0, shadowDepth = .2f, antiHalo = .901f, flatAreaProtection = 0;
    float tone = 0, exposureEV = 1, contrast = 1, saturation = 1, highlightCompression = 0;
};

struct RtgiSettings
{
    bool enabled = false;
    UINT quality = 2, denoiser = 1, inspect = 0;
    float mix = 1, lighting = 5, occlusion = 1, ambient = 1;
    float thickness = .1f, smoothness = .5f, fade = .3f, fov = 60, farPlane = 600;
    float contact = 0, saturation = 1, radius = 1;
    bool operator==(const RtgiSettings&) const = default;
};

inline float BoundedToneLift(float value)
{
    return std::isfinite(value) ? std::clamp(value, 0.0f, 0.25f) : 0.0f;
}

struct Settings
{
    UINT encoding = 0; // Auto, Linear, sRGB, Gamma 2.2
    bool everyFrame = false;
    // How many frames may be running at the NR stage at once. Too few and a frame that finds
    // every buffer busy carries no NR at all.
    UINT slots = 3;
    bool toneChannels = false;
    bool autoMask = true, useGameExposure = true;
    UINT style = 0, toneCurve = 0;
    float toneLift = 0;
    float modelScale = 1;
    UINT passes = 1;
    // Graphics preference from the bridge; each Record still requires admission.
    int spinDraw = 0;
    float tone = 0, structure = 1, skin = 1;
    LookSettings look;
    RtgiSettings rtgi;
    bool SameHistorySettings(const Settings& other) const
    {
        return encoding == other.encoding && toneChannels == other.toneChannels &&
               modelScale == other.modelScale && tone == other.tone && structure == other.structure &&
               skin == other.skin && everyFrame == other.everyFrame && autoMask == other.autoMask &&
               useGameExposure == other.useGameExposure && style == other.style &&
               toneCurve == other.toneCurve && BoundedToneLift(toneLift) == BoundedToneLift(other.toneLift);
    }
};

// Process lifetime owner: intentionally not destroyed/unloaded while worker threads exist.
class Backend
{
    struct Impl;
    Impl* p;

  public:
    Backend(ID3D12Device*, ID3D12CommandQueue*, const std::filesystem::path& directory);
    // Records pre-SR work. Returns an input for the upscaler, or nullptr on skip/failure.
    ID3D12Resource* Record(ID3D12GraphicsCommandList*, const Frame&, const Settings&);
    // Bind the render queue and enqueue a migration dependency BEFORE Execute.
    // Required for every submission; does not publish work yet.
    int PendingListIndex(UINT, ID3D12CommandList* const*) const;
    void Submitting(ID3D12CommandQueue*, UINT, ID3D12CommandList* const*);
    // Diagnostic snapshot only; does not flush, cancel, or destroy GPU work.
    void TraceBoundary(const std::string&);
    // Must run immediately AFTER real queue submission, including non-upscale lists.
    void Submitted(ID3D12CommandQueue*, UINT, ID3D12CommandList* const*);
    bool Ready();
    bool Shutdown(); // call before loader-lock teardown, after all submissions
    void ReleaseSession();
    bool PollRelease();
    // Finish a release deferred by a busy GPU. Returns false when still busy.
    bool CompletePendingReleaseLocked();
    void ResetGraphicsWaitState();
    void InvalidateHistory(); // applied at the next safe recording boundary
    std::string Status() const;
    bool IsRunning() const;
    UINT64 RecordedFrames() const;
};

// Runs the pass over Color immediately before Super Resolution consumes it. Safe to call every
// frame; it builds what it needs on first use and disables itself for the session if anything
// fails, rather than retrying into a crash. Color is returned in its original readable state.
// timingQueue is the queue this command list will be executed on, when the caller knows it --
// State::currentCommandQueue only exists once a D3D12 swapchain has been created, so without
// it the pass runs and never reports what it cost.
void EvaluateBeforeUpscale(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params,
                           ID3D12CommandQueue* timingQueue = nullptr);

void Shutdown();
bool IsRunning();
// Why it is not, if it is not. Empty while it is running or has not been tried yet.
const char* FailureReason();
// Clears the session failure latch, so a failure caused by transient thrash does not cost a
// restart.
void RetryAfterFailure();
// The settings panel, drawn inside OptiScaler's menu. Not wired up yet: there is nothing to
// adjust until the backend has settings of its own.
void RenderMenu(::Config* config, float menuResScale);
} // namespace NrBridge
