/* 
 * static_for_loop.h                                                               
 * 
 * L. Mazur 
 * 
 * Sometimes we need to loop over template parameters. Unfortunately, template parameters have to be known at compile time.
 * That makes it impossible to simple use the c++14 for-loop, because the loop variable has to be dynamic...
 * C++20 supports a way of looping over such a thing, but we are not using C++20 yet..
 * That's why we need this ugly workaround (ref: https://stackoverflow.com/questions/13816850/is-it-possible-to-develop-static-for-loop-in-c ).
 *
 * To use it, do the following:
 * 
 * static_for<0, Channel>::apply([&](auto i) // Changed from '(int i)'. In general, 'auto' will be a better choice for meta-programming!
 * {            
 *     // code...
 *     do_something_with<i>()
 * });
 * 
 */

#ifndef STATIC_FOR_LOOP_H
#define STATIC_FOR_LOOP_H


template <int First, int Last>
struct static_for
{
    template <typename Lambda>
        static inline constexpr void apply(__attribute__((unused)) Lambda const& f)
        {
            if (First < Last)
            {
                f(std::integral_constant<int, First>{});
                static_for<First + 1, Last>::apply(f);
            }
        }
};
template <int N>
struct static_for<N, N>
{
    template <typename Lambda>
        static inline constexpr void apply(__attribute__((unused)) Lambda const& f) {}
};


#endif //STATIC_FOR_LOOP_H
