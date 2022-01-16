/* 
 * main_memManTest.cpp                                                               
 * 
 * D. Clarke
 * 
 * Test the methods of the new memory management, both for the device and the host.
 * 
 */

#include "../SIMULATeQCD.h"

#define PREC double
#define MY_BLOCKSIZE 256

gMemoryPtr<true> createRvalueRef(gMemoryPtr<true> mem){
    rootLogger.info("Create Rvalue ref...");
    return mem;
}
gMemoryPtr<false> createRvalueRef(gMemoryPtr<false> mem){
    rootLogger.info("Create Rvalue ref...");
    return mem;
}

template<bool onDevice>
bool run(){
    bool passed=true;

    gMemoryPtr<onDevice> mem_1 = MemoryManagement::getMemAt<onDevice>("abc");
    gMemoryPtr<onDevice> shr_1 = MemoryManagement::getMemAt<onDevice>("SHARED_abc");
    rootLogger.info("Should be 0:    " ,  mem_1->getSize());
    if(mem_1->getSize()!=0) passed=false;
    rootLogger.info("Should be 0:    " ,  shr_1->getSize());
    if(shr_1->getSize()!=0) passed=false;
    mem_1->adjustSize(1024);
    shr_1->adjustSize(1024);
    rootLogger.info("Should be 1024: " ,  mem_1->getSize());
    if(mem_1->getSize()!=1024) passed=false;
    rootLogger.info("Should be 1024: " ,  shr_1->getSize());
    if(shr_1->getSize()!=1024) passed=false;
    MemoryManagement::memorySummary();

    {
        /// This should return 0 because this section should know about "abc", and SmartNames don't allow for the same
        /// name to identify the same gMemoryPtr if it's unique.
        gMemoryPtr<onDevice> mem_scope_1 = MemoryManagement::getMemAt<onDevice>("abc");
        rootLogger.info("Should be 0:    " ,  mem_scope_1->getSize());
        if(mem_scope_1->getSize()!=0) passed=false;
        /// But not this one because it's shared.
        gMemoryPtr<onDevice> shr_scope_1 = MemoryManagement::getMemAt<onDevice>("SHARED_abc");
        rootLogger.info("Should be 1024: " ,  shr_scope_1->getSize());
        if(shr_scope_1->getSize()!=1024) passed=false;
        gMemoryPtr<onDevice> mem_scope_2 = MemoryManagement::getMemAt<onDevice>("def");
        mem_scope_2->adjustSize(512);
        rootLogger.info("Should be 512:  " ,  mem_scope_2->getSize());
        if(mem_scope_2->getSize()!=512) passed=false;
        MemoryManagement::memorySummary();
    }
    MemoryManagement::memorySummary();

    /// Again should return 0 because "abc" was already declared.
    gMemoryPtr<onDevice> mem_2 = MemoryManagement::getMemAt<onDevice>("abc");
    rootLogger.info("Should be 0:    " ,  mem_2->getSize());
    if(mem_2->getSize()!=0) passed=false;
    /// And again, this one should give 1024 because it shares with mem_1.
    gMemoryPtr<onDevice> shr_2 = MemoryManagement::getMemAt<onDevice>("SHARED_abc");
    rootLogger.info("Should be 1024: " ,  shr_2->getSize());
    if(shr_2->getSize()!=1024) passed=false;
    /// This should return 0 because "def" went out of scope
    gMemoryPtr<onDevice> mem_3 = MemoryManagement::getMemAt<onDevice>("def");
    rootLogger.info("Should be 0:    " ,  mem_3->getSize());
    if(mem_3->getSize()!=0) passed=false;
    mem_3->adjustSize(256);
    rootLogger.info("Should be 256:  " ,  mem_3->getSize());
    if(mem_3->getSize()!=256) passed=false;
    /// mem_2 and mem_1 are sharing memory, so this should have affected it.
    shr_2->adjustSize(2048);
    rootLogger.info("Should be 2048: " ,  shr_2->getSize());
    if(shr_2->getSize()!=2048) passed=false;
    rootLogger.info("Should be 2048: " ,  shr_1->getSize());
    if(shr_1->getSize()!=2048) passed=false;

    /// Some copy tests.
    gMemoryPtr<onDevice> cp_1 = MemoryManagement::getMemAt<onDevice>("mno");
    gMemoryPtr<onDevice> cp_2 = MemoryManagement::getMemAt<onDevice>("pqr");
    cp_1->adjustSize(987);
    cp_2->adjustSize(678);
    MemoryManagement::memorySummary();
    cp_1 = cp_2;
    MemoryManagement::memorySummary();
    rootLogger.info("Should be 678:  " ,  cp_1->getSize());
    if(cp_1->getSize()!=678) passed=false;

    /// Some move/copy tests (only different from code above if the flag in MemoryManagement.h is set).
    gMemoryPtr<onDevice> mv_1 = createRvalueRef(MemoryManagement::getMemAt<onDevice>("ghi"));
    gMemoryPtr<onDevice> mv_2 = createRvalueRef(MemoryManagement::getMemAt<onDevice>("jkl"));
    mv_1->adjustSize(987);
    mv_2->adjustSize(678);
    mv_1 = mv_2;
    rootLogger.info("Should be 678:  " ,  mv_1->getSize());
    if(mv_1->getSize()!=678) passed=false;

    MemoryManagement::memorySummary();
    return passed;
}


int main() {

    rootLogger.setVerbosity(TRACE);

    ///-------------------------------------------------------------------------------------------------------HOST TESTS
    rootLogger.info("oooooooooooooooooooo");
    rootLogger.info("o BEGIN HOST TESTS o");
    rootLogger.info("oooooooooooooooooooo");

    bool passed_host = run<false>();
    if(!passed_host) rootLogger.error("Host test failed!");

    ///-----------------------------------------------------------------------------------------------------DEVICE TESTS
    rootLogger.info("oooooooooooooooooooooo");
    rootLogger.info("o BEGIN DEVICE TESTS o");
    rootLogger.info("oooooooooooooooooooooo");

    bool passed_dev = run<true>();
    if(!passed_dev) rootLogger.error("Device test failed!");

    if(passed_host&&passed_dev) {
        rootLogger.info("All tests " ,  CoutColors::green ,  "passed!" ,  CoutColors::reset);
    } else {
        rootLogger.error("At least one test failed!");
        return -1;
    }

    return 0;
}

