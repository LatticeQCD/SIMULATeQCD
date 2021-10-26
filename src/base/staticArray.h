#include <iostream>
#ifndef STATIC_ARRAY_H
#define STATIC_ARRAY_H

template<class T, size_t N>

class StaticArray{
    StaticArray<T, N - 1> rest;
    T my_object;


    public:

    T* begin(){
        return &rest[0];
    }

    T* end(){
        return &my_object + 1;
    }

    T& operator[](size_t i){
        if (i == N - 1){
            return my_object;
        }else{
            return rest[i];
        }
    }
    template<typename... Args>
    StaticArray(Args&&... args):rest(args...), my_object(args...){
        static_assert(N < 256, "The StaticArray class should not be used for large arrays");
    }
};



template<class T>
class StaticArray<T, 1>{
    T my_object;
    public:

    T* begin(){
        return &my_object;
    }

    T* end(){
        return &my_object + 1;
    }

    T& operator[]( [[gnu::unused]] size_t i){
        return my_object;
    }

    template<typename... Args>
    StaticArray(Args&&... args): my_object(args...){
    }
};





template<class T>
class StaticArray<T, 0>{
    public:

    T* begin(){
        return nullptr;
    }

    T* end(){
        return nullptr;
    }

    T& operator[]( [[gnu::unused]] size_t i){
        throw PGCError("Size of array is 0");
    }

    template<typename... Args>
    StaticArray(__attribute__((unused)) Args&&... args){}
};

#endif
