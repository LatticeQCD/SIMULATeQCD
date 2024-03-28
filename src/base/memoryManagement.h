/*
 * memoryManagement.h
 *
 * L. Mazur, D. Clarke, L. Altenkort
 *
 * This header file includes everything needed to manage dynamically allocated memory in the code. The idea is as
 * follows: Each instance of dynamic memory is implemented as a gMemory object. A gMemory object contains the raw
 * pointer and the size in bytes of the allocated memory, as well as wrappers for CUDA functions needed to allocate
 * memory on GPUs, and functions for copying, swapping, and resizing.
 *
 * The MemoryManagement class is the only thing in the code allowed to manipulate gMemory objects directly, and WE
 * SHOULD STRIVE NOT TO USE ANY OTHER KIND OF DYNAMICALLY ALLOCATED MEMORY WHEN WE CODE. In this way, MemoryManagement
 * knows about all dynamically allocated memory, and is the only thing allowed to directly mess with it.
 *
 * Within the MemoryManager, gMemory objects are stored in Containers, which are implemented through std::map. Using
 * std::map we associate to each gMemory object a name (std::string). (For those of you familiar with python, we
 * essentially have a dictionary where they keys are the names and the values are the gMemory objects.) There are
 * two kinds of Containers: device containers and host containers.
 *
 * Implemented in this header are also gMemoryPtr objects, which you should be able to use just like real pointers.
 * gMemoryPtrs are special in that they comply with all this name/container functionality.
 *
 * When you wish to allocate dynamic memory, use MemoryManagement::getMemAt(name). If name begins with "SHARED_" it
 * will behave as channels did before. Otherwise, the MemoryManager enforces that if getMemAt(name) is called again
 * later, it will get its own memory, separate from the first time getMemAt(name) was called. For some very basic
 * examples how this works, please read through main_memManTest.cpp.
 *
 */

#pragma once
#include "wrapper/gpu_wrapper.h"
#include "../define.h"
#include "gutils.h"
#include <cstdlib>
#include <utility>
#include <vector>
#include <memory>
#include <cstring>
#include <map>
#include "./communication/gpuIPC.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

#define DISALLOW_SHARED_MEM 0     /// True: Enforce that all memory channels are unique

/// This line is needed because the MemoryManagement and MemoryAccessor need to know what gMemoryPtr is, and compilation
/// happens in top-bottom order. We will define what gMemoryPtr is later. We unfortunately can't define it here because
/// the gMemoryPtr class needs to use some functions from the MemoryManagement. Look at this tangle of thorns.
template<bool onDevice>
class gMemoryPtr;


/// This is the MemoryManagement class. It manages all instances of dynamic memory in the code. Every important member
/// is implemented as static, because we would like to avoid instantiating the MemoryManager at the beginning of every
/// program. This way, the MemoryManager is always in the background.
class MemoryManagement {

//! The first thing in here is a private class, namely gMemory, inside of a class! This is to prevent access from outside.
private:
    //! gMemory objects contain the actual pointers to dynamic memory. It should only be possible to construct them through
    //! the creation of a gMemoryPtr object, which in turn can only be constructed by the MemoryManagement.
    template<bool onDevice>
    class gMemory {
    private:
        size_t _current_size; /// Size of rawPointer in bytes.
        void *_rawPointer;    /// Declare a rawPointer pointer of type void, which we want to do because void type pointers
        /// can point to an object of any type. A goal of the new memory management is that this
        /// will be the only raw pointer.

        /// Wrapper for allocating memory using CUDA built-in functions. For some reason, gpuMalloc takes a pointer to
        /// a pointer as an argument; hence the strange syntax.
        void alloc(size_t size) {
            if (size > 0) {
                if (onDevice) {
                    gpuError_t gpuErr = gpuMalloc((void **) &_rawPointer, size);
                    if (gpuErr != gpuSuccess) {
                        MemoryManagement::memorySummary(false, false, true, true,false);
                        std::stringstream err_msg;
                        err_msg << "_rawPointer: Failed to allocate (additional) " << size/1000000000. << " GB of memory on device";
                        GpuError(err_msg.str().c_str(), gpuErr);
                    }
                } else {
#ifndef CPUONLY
                    gpuError_t gpuErr = gpuMallocHost((void **) &_rawPointer, size);
                    if (gpuErr != gpuSuccess) {
                        MemoryManagement::memorySummary(true,true,false, false,false);
                        std::stringstream err_msg;
                        err_msg << "_rawPointer: Failed to allocate (additional) " << size/1000000000. << " GB of memory on host";
                        GpuError(err_msg.str().c_str(), gpuErr);
                    }
#else
                    _rawPointer = std::malloc(size);
                    if (_rawPointer == nullptr){
                        MemoryManagement::memorySummary(true,true,false, false,false);
                        std::stringstream err_msg;
                        err_msg << "_rawPointer: Failed to allocate (additional) " << size/1000000000. << " GB of memory on host";
                        throw std::runtime_error(stdLogger.fatal(err_msg.str()));
                    }
#endif
                }
                rootLogger.alloc("> Allocated mem at " ,  static_cast<void*>(_rawPointer)
                                   ,  " (" ,  (onDevice ? "Device" : "Host  ") ,  "): "
                                   ,  size/1000000000. ,  " GB");
            }
            _current_size = size;
        }

        /// Wrapper for freeing memory using CUDA built-in functions. By contrast with gpuMalloc, gpuFree takes just the
        /// pointer as the argument. No idea why, but I guess it doesn't matter.
        void free() {
            if (_current_size > 0) {
                if (onDevice) {
                    if (P2Ppaired) {
                        _cIpc.destroy();
                    }
                    gpuError_t gpuErr = gpuFree(_rawPointer);

                    if (gpuErr != gpuSuccess) {
                        MemoryManagement::memorySummary(false, false, true, true, false);
                        std::stringstream err_msg;
                        err_msg << "_rawPointer: Failed to free memory of size " << _current_size/1000000000. <<
                                "GB at " << static_cast<void*>(_rawPointer) << " on device";
                        GpuError(err_msg.str().c_str(), gpuErr);
                    }

                } else {
#ifndef CPUONLY
                    gpuError_t gpuErr = gpuFreeHost(_rawPointer);
                    if (gpuErr != gpuSuccess) {
                        MemoryManagement::memorySummary(true,true,false, false, false);
                        std::stringstream err_msg;
                        err_msg << "_rawPointer: Failed to free memory of size " << _current_size/1000000000. <<
                                "GB at " << static_cast<void*>(_rawPointer) << " on host";
                        GpuError(err_msg.str().c_str(), gpuErr);
                    }
#else
                    std::free(_rawPointer);
#endif
                }
                rootLogger.alloc("> Free      mem at " ,  static_cast<void*>(_rawPointer) ,  " (" ,  (onDevice ? "Device" : "Host  ") ,  "): " ,
                                   _current_size/1000000000. ,  " GB");
            }
            _rawPointer = nullptr;
            _current_size = 0;
        }



    public:
        //! gMemory constructor; initialize _current_size and _rawPointer.
        //! We should figure out how to make this private. Problem: befriending std::map or std::pair doesn't work.
        explicit gMemory(size_t size) : _current_size(size), _rawPointer(nullptr), P2Ppaired(false) {
            adjustSize(size);
        }

        /// First we remove the default copy constructors. We want to disallow any copies of a gMemory object, because if
        /// there were a copy, the MemoryManager would not know about it. The MemoryManager shall be omniscient. To
        /// understand the syntax, google references and R-value references.
        gMemory(const gMemory<onDevice> &) = delete;

        gMemory(const gMemory<onDevice> &&) = delete;

        gMemory(gMemory<onDevice> &) = delete;

        gMemory(gMemory<onDevice> &&) = delete;



        /// We make sure we free up memory automatically when we exit whatever scope a gMemory object is defined in.
        /// In general we should do this whenever we dynamically allocate memory. If we're not dynamically allocating
        /// memory we don't have to worry about it, because memory allocation is handled automatically by the stack.
        ~gMemory() {
            free(); /// Remember that this was defined above
        }

        void swap(gMemoryPtr<onDevice> &src);

        void memset(int value)
        {
            if (_current_size > 0) {
                if (onDevice) {
                    gpuError_t gpuErr = gpuMemset(_rawPointer, value, _current_size);
                    if (gpuErr != gpuSuccess) GpuError("_rawPointer: Failed to set memory on device", gpuErr);
                } else {
                    std::memset(_rawPointer, value, _current_size);
                }
            }
        }

        template<bool onDeviceSrc>
        void copyFrom(const gMemoryPtr<onDeviceSrc> &src, size_t sizeInBytes, size_t offsetSelf = 0,
                      size_t offsetSrc = 0);

        /// Assignment operator.
        template<bool onDeviceSrc>
        gMemory<onDevice> &operator=(const gMemory<onDeviceSrc> &memRHS) {
            copyFrom(memRHS, memRHS.getSize());
            return *this;
        }

        /// If we need more memory for the array pointed to by _rawPointer, allocate some more space. If adjustSize
        /// receives a template parameter, allocate an amount of space appropriate for the specified type. Otherwise
        /// allocate the size in bytes directly.
        template<class T>
        bool adjustSize(size_t size) {
            size_t sizeBytes = sizeof(T) * size;
            bool resize = _current_size < sizeBytes;
            if (resize) {
                if (_current_size > 0){
                    rootLogger.alloc("Increasing mem (" ,  (onDevice ? "Device" : "Host") ,  ") from " ,
                                       _current_size/1000000000. ,  " GB to " ,  sizeBytes/1000000000. ,  " GB:");
                }
                free();
                alloc(sizeBytes);
                if (P2Ppaired && onDevice) _cIpc.updateAllHandles(getPointer<uint8_t>());
            }
            return resize;
        }

        bool adjustSize(size_t sizeBytes) {
            bool resize = _current_size < sizeBytes;
            if (resize) {
                if (_current_size > 0){
                    rootLogger.alloc(">> Increase mem at " ,  static_cast<void*>(_rawPointer) ,  " (" ,  (onDevice ? "Device" : "Host") ,  ") from " ,
                                       _current_size/1000000000. ,  " GB to " ,  sizeBytes/1000000000. ,  " GB.");
                }
                free();
                alloc(sizeBytes);
                if (P2Ppaired && onDevice) _cIpc.updateAllHandles(getPointer<uint8_t>());

            }
            return resize;
        }

        /// Again, arithmetic on void pointers is not allowed, so convert to char. static_cast is compile time casting.
        /// Also remember that _rawPointer is the pointer to the block of memory that we're interested in; if we never
        /// required pointer arithmetic, we would just have written return _rawPointer. But we need it for stacked spinors.
        /// Implemented in cpp file because the compiler does not recognize the second instance as a pointer.
        void *getPointer(const size_t offsetInBytes = 0) const {
            return static_cast<char *>(_rawPointer) + offsetInBytes;
        }

        template<class T>
        __host__ __device__ T *getPointer(const size_t offsetInUnitsOfT = 0) const {
            return static_cast<T *>(_rawPointer) + offsetInUnitsOfT;
        }

        /// Returns size of buffer in bytes
        size_t getSize() const { return _current_size; }

        /// Overload the stream operator for use in memorySummary()
        friend std::ostream &operator<<(std::ostream &s, const std::map<std::string,gMemory<onDevice>>& container) {
            size_t total = 0;
            for ( auto it = container.begin(); it != container.end(); it++ ) {
                std::string name = it->first;
                size_t byteSize = it->second.getSize();
                total += byteSize;
                s << name << ": " << byteSize << " Bytes\n\t";
            }

            s << "Total: " << total << " Bytes (" << total/1000000000. << " GB)\n";
            return s;
        }



    /// THIS IS ALL CUDA IPC/P2P related stuff
    private:
    #if defined(USE_CUDA) || defined(USE_HIP)
        gpuIPC _cIpc;
        bool P2Ppaired;
    #endif
    public:
        void initP2P(MPI_Comm comm, int myRank) {
            if (onDevice && _current_size != 0) {
                if (!P2Ppaired) {
                    #if defined(USE_CUDA) || defined(USE_HIP)
                    _cIpc = gpuIPC(comm, myRank, this->template getPointer<uint8_t>());
                    P2Ppaired = true;
                    #endif
                }
            } else if (!onDevice) {
                rootLogger.error("memoryManagement.h: initP2P: gpuIPC not possible on gMemory<HOST>");
            }
        }

        void addP2PRank(int oppositeRank) {
            #if defined(USE_CUDA) || defined(USE_HIP)
            if (onDevice) _cIpc.addP2PRank(oppositeRank);
            else if (!onDevice) {
                rootLogger.error("memoryManagement.h: addP2PRank: gpuIPC not possible on gMemory<HOST>");
            }
            #endif
        }

        void syncAndInitP2PRanks() {
            #if defined(USE_CUDA) || defined(USE_HIP)
            if (onDevice && _current_size != 0) _cIpc.syncAndInitAllP2PRanks();
            else if (!onDevice) {
                rootLogger.error("memoryManagement.h: syncAndInitP2PRanks: gpuIPC not possible on gMemory<HOST>");
            }
            #endif
        }

        uint8_t *getOppositeP2PPointer(int oppositeRank) {
            #if defined(USE_CUDA) || defined(USE_HIP)
            if (onDevice && _current_size != 0) return _cIpc.getPointer(oppositeRank);
            else if (!onDevice) {
                rootLogger.error("memoryManagement.h: getOppositeP2PPointer: gpuIPC not possible on gMemory<HOST>");
                return nullptr;
            }
            #endif
            return nullptr;
        }
    };



private:
    //! The gMemoryPtr needs access to increase and decrease the Counts
    template<bool onDevice> friend class gMemoryPtr;
    friend class MemoryAccessor;
private:
    /// This is where we will keep our gMemory objects and the count of gMemoryPtr's that point to them.
    static std::map<std::string, gMemory<true>> devContainer;
    static std::map<std::string, gMemory<false>> hostContainer;
    static std::map<std::string, size_t> devCounts;
    static std::map<std::string, size_t> hostCounts;

private:
    /// The following are some functions needed to look at the gMemoryPtr names. To implement unique gMemoryPtr names,
    /// we append _N at the end, where N is some integer that I call a tag. These methods allow one to extract either
    /// a base name or a tag from these strings.
    static bool is_number(const std::string &s) {
        std::string::const_iterator it = s.begin();
        while (it != s.end() && std::isdigit(*it)) ++it;
        return !s.empty() && it == s.end();
    }

    static int extract_tag(const std::string& name) {
        int position = name.find_last_of('_');
        std::string temp = name.substr(position + 1);
        if (!is_number(temp)) {
            throw std::runtime_error(stdLogger.fatal("Expected to extract integer from gMemoryPtr tag!"));
        }
        return std::stoi(temp);
    }

    static std::string extract_name(const std::string& name) {
        int position = name.find_last_of('_');
        return name.substr(0, position);
    }

    /// Look for any instances of a name, even as a substring, within the container.
    static bool is_substr_in_here(const std::string& substr, std::map<std::string, gMemory<true>> &container) {
        bool lstr = false;
        for (auto & it : container) {
            /// Must be recast as int because find is too stupid to know that -1 isn't a giant positive integer.
            if ((int) (it.first.find(substr)) >= 0) lstr = true;
        }
        return lstr;
    }

    /// Given a name, find the largest tag for that name.
    static int extract_max_tag_for_substr(const std::string& substr, std::map<std::string, gMemory<true>> &container) {
        int maxTag = 0;
        for (auto & it : container) {
            std::string temp = it.first;
            if (extract_name(substr) != extract_name(temp)) continue;
            if ((int) temp.find(substr) >= 0) maxTag = std::max(extract_tag(temp), maxTag);
        }
        if (maxTag < 0) {
            throw std::runtime_error(stdLogger.fatal("Something strange happened while extracting max tag for gMemoryPtr. substr, maxTag = ", substr, "  ", maxTag));
        }
        return maxTag;
    }

    /// Get a SmartName, i.e. a name with an appended tag in case the user isn't sure whether the memory should be
    /// shared or not. If multiple gMemoryPtrs are instantiated with the same name, and if the user didn't specify that
    /// this name should be shared, the SmartName will ensure that it creates a unique name from the original one.
    static std::string getSmartName(std::string name, std::map<std::string, gMemory<true>> &container) {
        /// True if the name should be unique, i.e. if it doesn't start with SHARED_ or if shared memory has
        /// been disallowed by the user.
        if (((int) name.find("SHARED_") < 0) || DISALLOW_SHARED_MEM) {
            if (!is_substr_in_here(name, container)) { /// True if name is not a substring anywhere in devContainer
                name.append("_0");
            } else {
                int largestTag = extract_max_tag_for_substr(name, container);
                largestTag++;
                name = extract_name(name).append(std::to_string(largestTag).insert(0, "_"));
            }
        }
        return name;
    }

    /// Same as above but for the host container.
    static bool is_substr_in_here(const std::string& substr, std::map<std::string, gMemory<false>> &container) {
        bool lstr = false;
        for (auto & it : container) {
            /// Must be recast as int because find is too stupid to know that -1 isn't a giant positive integer.
            if ((int) (it.first.find(substr)) >= 0) lstr = true;
        }
        return lstr;
    }

    static int extract_max_tag_for_substr(const std::string& substr, std::map<std::string, gMemory<false>> &container) {
        int maxTag = 0;
        for (auto & it : container) {
            std::string temp = it.first;
            if (extract_name(substr) != extract_name(temp)) continue;
            if ((int) temp.find(substr) >= 0) maxTag = std::max(extract_tag(temp), maxTag);
        }
        if (maxTag < 0) {
            throw std::runtime_error(stdLogger.fatal("Something strange happened while extracting max tag for gMemoryPtr. name, maxTag =", substr, "  ", maxTag));
        }
        return maxTag;
    }

    static std::string getSmartName(std::string name, std::map<std::string, gMemory<false>> &container) {
        if (((int) name.find("SHARED_") < 0) || DISALLOW_SHARED_MEM) {
            if (!is_substr_in_here(name, container)) {
                name.append("_0");
            } else {
                int largestTag = extract_max_tag_for_substr(name, container);
                largestTag++;
                name = extract_name(name).append(std::to_string(largestTag).insert(0, "_"));
            }
        }
        return name;
    }

private:
    static void decreaseDevCounter(const std::string& name);
    static void decreaseHostCounter(const std::string& name);
    static void increaseDevCounter(const std::string& name);
    static void increaseHostCounter(const std::string& name);
    static gMemoryPtr<true> getDevMem(std::string name, size_t size = 0);
    static gMemoryPtr<false> getHostMem(std::string name, size_t size = 0);

public:
    MemoryManagement() = default;

    //! Forbid copy or move
    MemoryManagement(const MemoryManagement &) = delete;
    MemoryManagement(const MemoryManagement &&) = delete;
    MemoryManagement(MemoryManagement &) = delete;
    MemoryManagement(MemoryManagement &&) = delete;



public:
    //! These are the only functions you may need from the outside:
    template<bool onDevice>
    static gMemoryPtr<onDevice> getMemAt(const std::string& name, size_t size = 0);
    static void memorySummary(bool show_counts_host=true, bool show_size_host=true,
                              bool show_counts_device=true, bool show_size_device=true, bool rootOnly=true);

};

/// Finally we can define the gMemoryPtr object. Such objects will have associated to them a name.
template<bool onDevice>
class gMemoryPtr {
private:
    /// It shouldn't be possible to modify the private members in here without the MemoryManagement.
    MemoryManagement::gMemory<onDevice> *raw;  /// Raw pointer to gMemory object.
    std::string name;

    /// Explicit constructor. Can only be called from MemoryManagement.
    gMemoryPtr(MemoryManagement::gMemory<onDevice>& raw, std::string name) : raw(&raw), name(name) {
        onDevice ? MemoryManagement::increaseDevCounter(name) : MemoryManagement::increaseHostCounter(name);
    }

public:

    friend class MemoryManagement;

    //! default constructor
    gMemoryPtr() : raw(nullptr), name("") {
    }

    //! copy constructor
    gMemoryPtr(const gMemoryPtr<onDevice> &source) : raw(source.raw), name(source.name) {
        onDevice ? MemoryManagement::increaseDevCounter(name) : MemoryManagement::increaseHostCounter(name);
    }

    //! copy assignment operator
    gMemoryPtr& operator=(const gMemoryPtr<onDevice>& source) {
        if ( &source != this ){
            if (onDevice) {
                MemoryManagement::decreaseDevCounter(name);
                name = source.name;
                raw = source.raw;
                MemoryManagement::increaseDevCounter(name);
            } else {
                MemoryManagement::decreaseHostCounter(name);
                name = source.name;
                raw = source.raw;
                MemoryManagement::increaseHostCounter(name);
            }
        }
        return *this;
    }

    //! Move constructor
    //! The total number of gMemoryPtr's to "name" stays the same when moving, so we don't have to increase or
    //! decrease the Counts here
    gMemoryPtr(gMemoryPtr<onDevice>&& source) noexcept  {
        name = source.name;
        raw = source.raw;
        source.name = "";
        source.raw = nullptr;
    }

    //! Move assignment operator
    //! The total number of gMemoryPtr's to "name" stays the same when moving, so we don't have to increase or
    //! decrease the Counts here
    gMemoryPtr& operator=(gMemoryPtr<onDevice> &&source)  noexcept {
        name = source.name;
        raw = source.raw;
        source.name = "";
        source.raw = nullptr;
        return *this;
    }

    //! This helps the gMemoryPtr behave like a real pointer.
    //! This way, by using the -> operator on a gMemoryPtr you're actually using it on the gMemory object it points to!
    //! (this is called "drill-down behavior")
    __host__ __device__ MemoryManagement::gMemory<onDevice>* operator->() const {
        return raw;
    }

    /// The pointer should free memory automatically when destroyed
    ~gMemoryPtr() {
        //! We only want to decrease the Count when an explicitly constructed gMemoryPtr (from the MemoryManagement
        //! with a non-empty name) is destructed.
        if (!name.empty()){
            onDevice ? MemoryManagement::decreaseDevCounter(name) : MemoryManagement::decreaseHostCounter(name);
        }
    }
};

class MemoryAccessor {
protected:
    void *Array; /// Again void type so we can allow for many different types of Array.

public:
    /// Constructors for MemoryAccessor class, overloaded.
    template<bool onDevice>
    explicit MemoryAccessor(MemoryManagement::gMemory<onDevice> reductionMemory) : Array(reductionMemory.template getPointer()) {}

    template<bool onDevice>
    explicit MemoryAccessor(gMemoryPtr<onDevice> reductionMemory) : Array(reductionMemory->template getPointer()) {}

    explicit MemoryAccessor(void *reductionArray) : Array(reductionArray) {}

    MemoryAccessor() : Array(nullptr) {}

    /// Destructor.
    ~MemoryAccessor() = default;

    template<class floatT>
    __device__  __host__ inline void setValue(const size_t isite, const floatT value) {
        /// reinterpret_cast is a compile time directive telling the compiler to treat _Array as a floatT*. This is
        /// needed because _Array is treated as void* right now.
        auto *arr = reinterpret_cast<floatT *>(Array);
        arr[isite] = value;
    }

    template<class floatT>
    __device__  __host__ inline void getValue(const size_t isite, floatT &value) {
        auto *arr = reinterpret_cast<floatT *>(Array);
        value = arr[isite];
    }

    template<class floatT>
    __device__ __host__ inline void getScalar(floatT &value) {
        auto *arr = reinterpret_cast<floatT *> (Array);
        value = *arr;
    }
};

