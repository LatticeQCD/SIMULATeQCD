/*
 * simulateqcd.h
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
#include "base/latticeContainer.h"
#include "base/latticeDimension.h"
#include "base/latticeParameters.h"
#include "base/memoryManagement.h"
#include "base/stopWatch.h"
#include "base/runFunctors.h"
#include "base/utilities/staticArray.h"
#include "base/utilities/static_for_loop.h"

#include "base/communication/communicationBase.h"
#include "base/communication/deviceEvent.h"
#include "base/communication/deviceStream.h"
#include "base/communication/gpuIPC.h"
  // Contains:
  //   base/communication/calcGSiteHalo_dynamic.h
#include "base/communication/haloOffsetInfo.h"
#include "base/communication/neighborInfo.h"
#include "base/communication/siteComm.h"

#include "base/indexer/bulkIndexer.h"
#include "base/indexer/haloIndexer.h"

#include "base/IO/fileWriter.h"
#include "base/IO/logging.h"
#include "base/IO/milc.h"
#include "base/IO/misc.h"
#include "base/IO/nersc.h"
#include "base/IO/evnersc.h"
#include "base/IO/parameterManagement.h"

#include "base/math/correlators.h"
#include "base/math/floatComparison.h"
#include "base/math/su3Accessor.h"
#include "base/math/su3Constructor.h"
#include "base/math/complex.h"
#include "base/math/generalAccessor.h"
#include "base/math/random.h"
#include "base/math/su2.h"
#include "base/math/su3array.h"
#include "base/math/su3.h"
#include "base/math/vect3array.h"
#include "base/math/vect3.h"
#include "base/math/matrix4x4.h"
#include "base/math/operators.h"
#include "base/math/simpleArray.h"
#include "base/math/stackedArray.h"
#include "base/math/su3Exp.h"

#include "base/wrapper/gpu_wrapper.h"
#include "base/wrapper/marker.h"

/// --------------------------------------------------------------------------------------------------- GAUGE AND SPINOR

#include "gauge/gaugeActionDeriv.h"
#include "gauge/gaugeAction.h"
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
#include "gauge/constructs/plaqConstructs.h"
#include "gauge/constructs/projectU3.h"
#include "gauge/constructs/rectConstructs.h"

#include "spinor/spinorfield.h"
#include "spinor/eigenpairs.h"

