# Memory Management

 ```{admonition} TL;DR:
 :class: toggle

The only functions you need to call from the outside to handle dynamic memory are:

```C++
//! Obtain gMemoryPtr to dynamic memory based on the name:
template<bool onDevice> static gMemoryPtr<onDevice> MemoryManagement::getMemAt(const std::string& name, size_t size = 0);

//! Print a summary of the current state of the dynamic memory:
static void MemoryManagement::memorySummary();

//! Retrieve or modify the content of the dynamic memory allocation:
template<class floatT> __device__  __host__ inline void MemoryAccessor::setValue(const size_t isite, const floatT value);
template<class floatT> __device__  __host__ inline void MemoryAccessor::getValue(const size_t isite, floatT &value)

//! Manipulate the gMemory object itself via gMemoryPtr->... :
template<class T> bool MemoryManagement::gMemory::adjustSize(size_t size)
void MemoryManagement::gMemory::swap(gMemoryPtr<onDevice> &src);
void MemoryManagement::gMemory::memset(int value);
template<bool onDeviceSrc> void MemoryManagement::gMemory::copyFrom(const gMemoryPtr<onDeviceSrc> &src, size_t sizeInBytes, size_t offsetSelf = 0, size_t offsetSrc = 0);
bool MemoryManagement::gMemory::adjustSize(size_t sizeBytes);
size_t MemoryManagement::gMemory::getSize() const;
...
```


## How it works

The idea of the static `MemoryManagement` class is to have a central object that manipulates and knows about all dynamic memory allocated in the code. It works as follows: Each instance of dynamic memory is enclosed by a `gMemory` object and referenced to the user through one or more corresponding `gMemoryPtr` objects. The content of the dynamic memory can only be accessed through the `MemoryAccessor` static class, and the memory object itself can be manipulated through its public member functions which can be accessed by dereferencing the corresponding `gMemoryPtr`'s.
A `gMemory` object contains the raw pointer and the size in bytes of the allocated memory, as well as wrappers for CUDA functions needed to allocate memory on GPUs, and functions for copying, swapping, and resizing. From the outside you only interact with these objects via `gMemoryPtr`'s. The `MemoryManagement` class is the only thing in the code allowed to create `gMemory` objects directly, and _we should strive not to use any other kind of dynamically allocated memory when we code_. If we allocate our own dynamic memory independent of the `MemoryMangement`, then it does not know about it, which defeats part of the purpose.

Within the `MemoryManagement`, `gMemory` objects are stored in containers, which are implemented through `std::map`. Using `std::map` we associate to each `gMemory` object a  `name` (`std::string`). (For those of you familiar with python, we essentially have a dictionary where the keys are the names and the values are the `gMemory` objects.) There are separate containers for the device and host.

The `MemoryManagement` enforces that if `getMemAt(name)` is called a second time with the same `name`, that the returned `gMemoryPtr` will point to its own `gMemory`, separate from the first time `getMemAt(name)` was called. Internally the name is appended with a unique "tag" (just a number starting at zero). For some very basic examples how this works, please read through `main_memManTest.cu` in the `src/testing` folder and compile+run it to see the output of `MemoryManagement::memorySummary()`. 

If the name begins with "`SHARED_`" it will not append a tag and refer to the same memory everytime you call `getMemAt`. It will then only change the dynamic memory allocation when the size needs to be increased. Many dynamic memory allocations of the code base are shared by default (for example halo buffers and lattice containers). 

The `gMemoryPtr` objects let you interact with the `gMemory` of a specific name and can be used just like real pointers. `gMemoryPtr` objects are special in that they comply with all this name/container functionality. Again, you will never interact with a `gMemory` object directly in your code; instead you will interact with (one of) its associated `gMemoryPtr`. The `MemoryManagement` keeps track of how many `gMemoryPtr` to any given `gMemory` are alive. Once every `gMemoryPtr` that points to a specific `gMemory` object is destroyed, the `gMemory` itself will be destroyed and the dynamic memory freed. In this way we don't need to keep track of the dynamic memory ourselves, and no memory leaks should occur. 

Here is an example of how you can allocate and manipulate dynamic memory on GPUs: 

```C++
/// Allocate some memory; call its pointer mem_1, and label it DescriptiveName
gMemoryPtr<true> mem_1 = MemoryManagement::getMemAt<true>("DescriptiveName");
std::cout << mem_1->getSize() << std::endl;

/// Change the size of the memory to which it points to 2024 bytes.
mem_1->adjustSize(2024); //! This calls a public method of the gMemory object
std::cout << mem_1->getSize() << std::endl;

/// Copy construct another gMemoryPtr:
gMemoryPtr<true> mem_2 = mem_1;
std::cout << mem_2->getSize() << std::endl;
```
The output will be as follows:
```
0
2024
2024
```


## Todo: add examples to show use of the MemoryAccessor etc
