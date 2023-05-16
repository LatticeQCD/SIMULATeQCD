/*
 * memoryManagement.cpp
 *
 * D. Clarke, L. Altenkort
 *
 * Implementations of some of the methods given in the MemoryManagement header file.
 *
 */

#include "memoryManagement.h"

/// Swap two chunks of memory by swapping their pointers.
template<bool onDevice>
void MemoryManagement::gMemory<onDevice>::swap(gMemoryPtr<onDevice> &src){
    if(getSize() == src->getSize()) {
        void *tmp = src->_rawPointer;
        src->_rawPointer = _rawPointer;
        _rawPointer = tmp;
    } else {
        rootLogger.error("memoryManagement.h: swap not allowed when sizes are different!");
    }
}

//! for documentation look at getHostMem
gMemoryPtr<true> MemoryManagement::getDevMem(std::string name, size_t size){
    if (name.empty()){
        throw std::runtime_error(stdLogger.fatal("MemoryManagement: Name for dynamic memory cannot be empty!"));
    }
    name = MemoryManagement::getSmartName(name,devContainer);
    if (devContainer.find(name) == devContainer.end()){
        devCounts.emplace(name, 0);
        devContainer.emplace(name, 0);
    }
    devContainer.at(name).adjustSize(size);
    return gMemoryPtr<true>(devContainer.at(name), name);
}

//! for documentation look at decreaseHostCounter
void MemoryManagement::decreaseDevCounter(const std::string& name){
    devCounts[name]--;
    if(devCounts[name] == 0){
        devContainer.erase(name);
        devCounts.erase(name);
    }
}

//! method to get a pointer to new dynamic memory, or just
gMemoryPtr<false> MemoryManagement::getHostMem(std::string name, size_t size){
    if (name.empty()){
        throw std::runtime_error(stdLogger.fatal("MemoryManagement: Name for dynamic memory cannot be empty!"));
    }

    //! convert the input string to a smart name
    name = MemoryManagement::getSmartName(name,hostContainer);

    //! if it doesn't exist yet, add it to the container(s)
    if (hostContainer.find(name) == hostContainer.end()){
        //! create a new pair of (string,size_t) that keeps track of the gMemoryPtr count
        hostCounts.emplace(name, 0);
        //! create a new pair of (string,gMemory) with size 0. In this line there is the only call to the explicit
        //! constructor of gMemory!
        hostContainer.emplace(name, 0);
    }

    //! allocate the dynamic memory using the pointer inside of the gMemory object.
    //! this doesn't do anything if size is smaller than the current size
    hostContainer.at(name).adjustSize(size);

    //! construct (another) gMemoryPtr that belongs to "name". the Count is increased in the explicit constructor.
    return gMemoryPtr<false>(hostContainer.at(name), name);
}

//! This function should only be called from the destructor of gMemoryPtr.
//! This function will free the dynamic memory associated with a gMemory object that is associated with a given name if
//! there are no more gMemoryPtr's that point to it. If there still are some, it will just decrease the Count by one.
void MemoryManagement::decreaseHostCounter(const std::string& name){
    //! this is the only place where the Count can be decreased.
    hostCounts[name]--;

    //! if there are no more gMemoryPtr that refer to "name", erase the corresponding gMemory object from the container.
    //! This calls the destructor of the gMemory object, which then frees the dynamic memory its member pointer
    //! points to.
    if(hostCounts[name] == 0){
        hostContainer.erase(name);
        hostCounts.erase(name);
    }
}

void MemoryManagement::increaseHostCounter(const std::string& name){
    hostCounts[name]++;
}
void MemoryManagement::increaseDevCounter(const std::string& name){
    devCounts[name]++;
}

/// Copy something into the chunk of memory. This method must be implemented in the cpp file because otherwise the
/// compiler doesn't recognize src->getPointer as a pointer for some reason.
template<bool onDevice>
template<bool onDeviceSrc>
void MemoryManagement::gMemory<onDevice>::copyFrom(const gMemoryPtr<onDeviceSrc> &src, const size_t sizeInBytes,
                                 const size_t offsetSelf , const size_t offsetSrc)
{
    adjustSize(sizeInBytes);
    gpuError_t gpuErr;
    if (onDevice) {
        /// Device to device
        if (onDeviceSrc) {
            /// Arithmetic on void pointers is not allowed. Therefore convert to char.
            gpuErr = gpuMemcpy(static_cast<char*>(_rawPointer) + offsetSelf,src->getPointer(offsetSrc),
                                 sizeInBytes, gpuMemcpyDeviceToDevice);
            if (gpuErr)
                GpuError("memoryManagement.h: Failed to copy data (DeviceToDevice)", gpuErr);
            /// Host to device
        } else {
            gpuErr = gpuMemcpy(static_cast<char*>(_rawPointer) + offsetSelf, src->getPointer(offsetSrc),
                                 sizeInBytes, gpuMemcpyHostToDevice);
            if (gpuErr)
                GpuError("memoryManagement.h: Failed to copy data (HostToDevice)", gpuErr);
        }
    } else {
        /// Device to host
        if (onDeviceSrc) {
            gpuErr = gpuMemcpy(static_cast<char*>(_rawPointer) + offsetSelf, src->getPointer(offsetSrc),
                                 sizeInBytes, gpuMemcpyDeviceToHost);
            if (gpuErr)
                GpuError("memoryManagement.h: Failed to copy data (DeviceToHost)", gpuErr);
            /// Host to host
        } else {
            gpuErr = gpuMemcpy(static_cast<char*>(_rawPointer) + offsetSelf, src->getPointer(offsetSrc),
                                 sizeInBytes, gpuMemcpyHostToHost);
            if (gpuErr)
                GpuError("memoryManagement.h: Failed to copy data (HostToHost)", gpuErr);
        }
    }
}



/// A convenient wrapper for getting host and device memory.
template<>
gMemoryPtr<true> MemoryManagement::getMemAt(const std::string& name, const size_t size) {
    return getDevMem(name, size);
}
template<>
gMemoryPtr<false> MemoryManagement::getMemAt(const std::string& name, const size_t size) {
    return getHostMem(name, size);
}

/// Overload stream operators

std::ostream &operator<<(std::ostream &s, const std::map<std::string,size_t>& container) {
    for (const auto & it : container) {
        std::string name = it.first;
        size_t counts    = it.second;
        s << name << ": " << counts << "\n\t";
    }
    return s;
}

std::map<std::string,MemoryManagement::gMemory<true>> MemoryManagement::devContainer;
std::map<std::string,size_t> MemoryManagement::devCounts;
std::map<std::string,MemoryManagement::gMemory<false>> MemoryManagement::hostContainer;
std::map<std::string,size_t> MemoryManagement::hostCounts;

/// Easy access to the state of the MemoryManagement for the purpose of debugging.
void MemoryManagement::memorySummary(bool show_counts_host, bool show_size_host,
                                     bool show_counts_device, bool show_size_device, bool rootOnly) {
     std::stringstream output;
     output << "MemoryManagement::memorySummary():\n";
    if (show_size_host || show_counts_host) {
        output << "\n# >>> HOST CONTAINER <<<\n";
        if (show_counts_host) {
            output << "\t---------------------------------\n"
                   << "\tname: ptr count\n"
                   << "\t---------------------------------\n"
                   << "\t" << hostCounts << "\n";
        }
        if (show_size_host) {
            output << "\t---------------------------------\n"
                   << "\tname: size\n"
                   << "\t---------------------------------\n"
                   << "\t" << hostContainer << "\n";
        }
    }
    if (show_size_device || show_counts_device){
        output << "# >>> DEVICE CONTAINER <<<\n";
        if (show_counts_device) {
            output << "\t---------------------------------\n"
                   << "\tname: ptr count\n"
                   << "\t---------------------------------\n"
                   << "\t" << devCounts << "\n";
        }
        if (show_size_device) {
            output << "\t---------------------------------\n"
                   << "\tname: size\n"
                   << "\t---------------------------------\n"
                   << "\t" << devContainer;
        }
    }
    if(rootOnly){
        rootLogger.info(output.str());
    }
    else{
        stdLogger.info(output.str());
    }
}



/// Again we have to instantiate all template possibilities.
template class MemoryManagement::gMemory<true>;
template class MemoryManagement::gMemory<false>;
template class gMemoryPtr<true>;
template class gMemoryPtr<false>;
template void MemoryManagement::gMemory<false>::copyFrom<true>(const gMemoryPtr<true> &src, const size_t sizeInBytes,
                                             const size_t offsetSelf, const size_t offsetSrc);
template void MemoryManagement::gMemory<false>::copyFrom<false>(const gMemoryPtr<false> &src, const size_t sizeInBytes,
                                              const size_t offsetSelf, const size_t offsetSrc);
template void MemoryManagement::gMemory<true>::copyFrom<false>(const gMemoryPtr<false> &src, const size_t sizeInBytes,
                                             const size_t offsetSelf, const size_t offsetSrc);
template void MemoryManagement::gMemory<true>::copyFrom<true>(const gMemoryPtr<true> &src, const size_t sizeInBytes,
                                            const size_t offsetSelf, const size_t offsetSrc);
