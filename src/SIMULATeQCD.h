/*
 * SIMULATeQCD.h
 *
 * Principal header file containing base, spinor, and gauge classes.
 *
 */

#pragma once

#include "define.h"
#include "explicit_instantiation_macros.h"
#include "preprocessorWrapper.h"

/// --------------------------------------------------------------------------------------------------------------- BASE

#include "base/gutils.h"
#include "base/LatticeContainer.h"
#include "base/LatticeDimension.h"
#include "base/latticeParameters.h"
#include "base/memoryManagement.h"
#include "base/stopWatch.h"
#include "base/runFunctors.h"
#include "base/staticArray.h"
#include "base/static_for_loop.h"

#include "base/communication/communicationBase.h"
#include "base/communication/deviceEvent.h"
#include "base/communication/deviceStream.h"
#include "base/communication/gpuIPC.h"
  // Contains:
  //   base/communication/calcGSiteHalo_dynamic.h
  //   base/communication/calcGSiteHalo.h
#include "base/communication/haloOffsetInfo.h"
#include "base/communication/neighborInfo.h"
#include "base/communication/siteComm.h"

#include "base/indexer/BulkIndexer.h"
#include "base/indexer/HaloIndexer.h"

#include "base/IO/fileWriter.h"
#include "base/IO/logging.h"
#include "base/IO/milc.h"
#include "base/IO/misc.h"
#include "base/IO/nersc.h"
#include "base/IO/parameterManagement.h"

#include "base/math/correlators.h"
#include "base/math/floatComparison.h"
#include "base/math/gaugeAccessor.h"
#include "base/math/gaugeConstructor.h"
#include "base/math/gcomplex.h"
#include "base/math/generalAccessor.h"
#include "base/math/grnd.h"
#include "base/math/gsu2.h"
#include "base/math/gsu3array.h"
#include "base/math/gsu3.h"
#include "base/math/gvect3array.h"
#include "base/math/gvect3.h"
#include "base/math/matrix4x4.h"
#include "base/math/operators.h"
#include "base/math/simpleArray.h"
#include "base/math/stackedArray.h"
#include "base/math/su3Exp.h"

#include "base/wrapper/gpu_wrapper.h"

/// --------------------------------------------------------------------------------------------------- GAUGE AND SPINOR

#include "gauge/gaugeActionDeriv.h"
#include "gauge/GaugeAction.h"
#include "gauge/gaugefield.h"

#include "gauge/constructs/derivative3link.h"
#include "gauge/constructs/derivative5link.h"
#include "gauge/constructs/derivative7link.h"
#include "gauge/constructs/derivativeLepagelink.h"
#include "gauge/constructs/derivativeProjectU3.h"
#include "gauge/constructs/fat7LinkConstructs.h"
#include "gauge/constructs/gsvd.h"
#include "gauge/constructs/hisqForceConstructs.h"
#include "gauge/constructs/naikDerivativeConstructs.h"
#include "gauge/constructs/PlaqConstructs.h"
#include "gauge/constructs/projectU3.h"
#include "gauge/constructs/RectConstructs.h"

#include "spinor/spinorfield.h"

