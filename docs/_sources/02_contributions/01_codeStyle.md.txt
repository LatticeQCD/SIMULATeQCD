# How to write clean and readable code

- **Warnings**\
Switch on compiler warnings with -Wall -Wextra. Do care about this warnings. A code should compile without any warning. If a warning is unavoidable use pragmas to suppress them.
- **Do not repeat yourself**\
Do not copy code or use similar code all the time. Never. This makes it very hard to maintain the code.
- **Do not use the new and delete operator**\
The new and the delete operator are source for many errors, especially memory leaks. It is better to avoid them completely and use `std::vector` instead. Especially there is absolutely no reason to use new for small objects. (Most objects are small. Even the GPU-Memory classes are small: They only contain pointers to the memory. The allocated memory itself is not part of the class.) Use the initialization list to create class members instead.  
- **Avoid pointers**\
Avoid pointers where you can. Use references instead. Pointer arithmetic is difficult to read and pointers are not type save. This causes a lot of errors. There are cases, where pointers are necessary (GPU memory). Everywhere else they should be avoided completely.
- **Delete the copy constructor for classes which manage large amount of memory**\
Delete the copy constructor:
`myClass(myClass& ) = delete`
The default copy constructor does not copy memory that is not part of the class. However, it calls the destructor. This deletes the memory that the class points to. This can lead to bad memory accesses. (`C++11`: Implement a move constructor instead)
- **Only functions should be public**\
Do not use public member variables in classes
- **Use const whenever possible**\
The usage of const can really help to avoid undesired modification of objects.
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
- **Do not implement the ()-operator**\
It is hard to distinguish the () from a function. Just do not use it.
- **Formatting**\
Format your code in a nice way:
    - Do not use tabs. Use spaces instead
    - Maximum line length of 120 characters.
    - Maximum function length of 80 lines.
    - Use braces:
      ```C++
      for(int i = 0; i < 10; i++){
              std::cout << i << std::endl;
      }
      ```
      instead of 
      ```C++
      for(int i = 0; i < 10; i++)
              std::cout << i << std::endl;
      ```
    - Do not put multiple statements in one line:
      ```C++
      if (true) {
              std::cout << "True" << std::endl;
      }
      ```
      instead of 
      ```C++
      if (true) {std::cout << "True" << std::endl;}
      ```
- **Document your code**\
Give comments about complex sections. Trivial code should not be commented. Remove old comments!
- **Remove old features**\
If features are not needed any more, remove them (for example CUDA 1.1 stuff)
- **Use `C++11/14/17/20` features** (lambda expressions, move constructor, initializer_list,...)\
Read about them. They are nice!
