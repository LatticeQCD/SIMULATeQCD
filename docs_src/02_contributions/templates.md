# Template instantiation

In order to make our code as flexible as possible and to reduce code repetition, we make
extensive use of C++ templates. At the same time, we would like to put kernel implementations
into `cpp` files in order to reduce build times.
This requires explicitly instantiating all possible template parameter combinations,
which is itself repetitive, messy, and tedious.

In order to help streamline this process we have introduced explicit instantiation macros,
which help automate the instantiation of the most commonly used template parameters.
These macros are kept in the header file `explicit_instantiation_macros.h`. 
For example one can
instantiate a generic class `INIT_CLASS` for all possible precisions,
halo depths, and gauge field compression types using `INIT_PHC(INIT_CLASS)`.

When you want to implement your own kernel in a `cpp` file. You will therefore need
to make sure that you all the following:
1. You have an appropriate forward declaration in your header file.
2. You have instantiated all possible template parameter combinations at the bottom
   of your `cpp` file. Here the instantiation macros can assist you.
3. You have included your `cpp` file in `CMakeLists.txt` with all flags for your template
   parameters set as you desire. See also our [code structure](codeStructure.md) article.