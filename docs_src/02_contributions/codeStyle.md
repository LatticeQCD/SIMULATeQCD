# How to write clean and readable code

- **Warnings**\
Switch on compiler warnings with -Wall -Wextra. Do care about these warnings. A code should compile without any warning. If a warning is unavoidable use pragmas to suppress them.
- **Do not repeat yourself**\
Do not copy code or use similar code all the time. This makes it harder to maintain the code, for example by requiring the developer to make the same change in multiple places.
- **Do not use the new and delete operators**\
The `new` and the `delete` operators are a source for many errors, especially memory leaks. It is better to avoid them completely and use the [Memory Manager](memoryAllocation.md). There is absolutely no reason to use `new` for small objects. (Most objects are small. Even the GPU-Memory classes are small: They only contain pointers to the memory. The allocated memory itself is not part of the class.) Use the initialization list to create class members instead.  
- **Avoid pointers**\
Avoid pointers where you can. Use references instead. Pointer arithmetic is difficult to read and pointers are not type safe. This causes a lot of errors. There are cases where pointers are necessary (GPU memory). Everywhere else they should be avoided completely.
- **Delete the copy constructor for classes which manage large amounts of memory**\
Delete the copy constructor using
`myClass(myClass& ) = delete`.
The default copy constructor does not copy memory that is not part of the class. However, it calls the destructor. This deletes the memory that the class points to. This can lead to bad memory accesses. Implement a move constructor instead.
- **Only functions should be public**\
Do not use public member variables in classes.
- **Use const whenever possible**\
The usage of `const` can help avoid undesired modification of objects.
- **Proper naming**\
Use proper and consistent naming for everything. Use the same name for file names as for the classes they contain. Use long names which describe what is done. Do not use different names between different function calls. Follow the following naming scheme:
    Classes:
        `ThisIsMyClass`
    Private class members:
        `_my_member`
    Member function:
        `void thisIsMyFunction()`
    Function parameter:
        `this_is_my_parameter`
        For example: `thisIsMyFunction(double this_is_my_parameter);`
- **Formatting**\
These are the conventions we have chosen for SIMULATeQCD. Please stick to them. Besides making the code a bit more readable, it will look more professional if we all follow the same conventions.
    - DO NOT USE TABS. Instead, please indent using 4 spaces.
    - Try for a maximum line length of 120 characters.
    - Try for a maximum function length of 80 lines.
    - Use braces:
      ```C++
      for(int i = 0; i < 10; i++) {
          ...
      }
      ```
      instead of 
      ```C++
      for(int i = 0; i < 10; i++)
          ...
      ```
    - Do not put multiple statements in one line:
      ```C++
      if (true) {
          ...
      }
      ```
      instead of 
      ```C++
      if (true) { ... }
      ```
    - Please format your if-else statements like so:
      ```C++
      if (true) {
          ...
      } else {
          ...
      }
      ```
- **Write helpful comments**
    - Give comments about complex sections, tricks that you used, relevant visualizations, DOIs for papers, your design philosophy (when relevant), etc. DO NOT comment trivial code. Remove commented-out code. (It is anyway in the git history if we need it later.)
    - Use `SIMULATeQCD/scripts/comments.bash` to put some leading comments at the top of your code. This helps ensure that we all include the same information in the same way at the beginning of our code.
- **Remove old features**\
If features are not needed any more, remove them. 
- **Use `C++11/14/17/20` features** (lambda expressions, move constructor, initializer_list,...)\
Read about them. They are nice!
