#pragma once

// The one declaration the reference's LmxxfEvaluateCut.h asks for, quoted from
// refs/optiscaler/ta_Selector.h:
//
//     // Startup hook policy: lmxxf, or convenience with lmxxf installed.
//     bool SubmissionHooksWanted();
//
// In the reference that policy is "lmxxf selected, or convenience with lmxxf installed", and
// the rest of ta_Selector.h is that project's own install probing (Kind, HasLmxxfInstalled,
// HasMochizukiInstalled, ...), none of which exists here: there is one backend, ours, and it
// has no runtime module to probe for. What is kept is the policy call itself, answered from
// this host's own setting for it, so LmxxfEvaluateCut.h compiles unchanged.
//
// It is a startup policy, which is why it is answered from a setting and not from anything
// measured: the reference latches it at startup and requires a restart after changing it.
#include "Config.h"

namespace DlssNr::Backend
{
inline bool SubmissionHooksWanted()
{
    return Config::Instance()->NrWrapLists.value_or_default();
}
} // namespace DlssNr::Backend
