#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "io_utils.hpp"

class CudaTimer {

    // A RAII wrapper to measure the time 
    // between two events on the device timeline.
    
    private:
        cudaEvent_t start_event, stop_event;

    public:
       CudaTimer() {
            cudaCheck(cudaEventCreate(&start_event));
            cudaCheck(cudaEventCreate(&stop_event));
        }

        ~CudaTimer() {
            // cudaEvent_t is really an opaque pointer
            // to an internal memory structure managed
            // by the Nvidia driver. cudaEventDestroy
            // accepts this pointer by value and frees
            // its internal resources. 
            // CRITICAL: start_event and stop_event become
            // dangling pointers after this. But this is 
            // safe, since the object is about to die
            // anyway.
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
        }

        void start(cudaStream_t stream = 0) {
            // The CPU drops a timestamp into the device's 
            // default stream, and moves on immediately. 
            // When the GPU encounters the event in its stream
            // queue, it takes a nanosecond-precise timestamp
            // and stores it in the driver.
            // To be used right before the section of code to 
            // be timed.
            cudaCheck(cudaEventRecord(
                start_event,
                stream
            ));
        }

        float stop(cudaStream_t stream = 0) {
            
            // Again, drop a timestamp in the default stream.
            // To be used right after the section of code to
            // be timed.
            cudaCheck(cudaEventRecord(
                stop_event,
                stream
            ));

            // The CPU halts and waits for a device signal over 
            // the PCI-e bus. The signal arrives once the GPU
            // has completed all kernels/ops in the default stream
            // and pulls this event from its queue.
            cudaCheck(cudaEventSynchronize(stop_event));

            float milliseconds {0.0};
            cudaCheck(cudaEventElapsedTime(
                &milliseconds, 
                start_event,
                stop_event
            ));
            return milliseconds;
        }

};