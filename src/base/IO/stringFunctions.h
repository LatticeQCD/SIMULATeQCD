//
// Created by Lukas Mazur on 05.12.21.
//

#ifndef STRINGFUNCTIONS_H
#define STRINGFUNCTIONS_H

#include <iostream>
#include <sstream>

namespace COLORS {
    const std::string red("\033[0;31m");
    const std::string redBold("\033[1;31m");
    const std::string green("\033[0;32m");
    const std::string greenBold("\033[1;32m");
    const std::string yellow("\033[0;33m");
    const std::string yellowBold("\033[1;33m");
    const std::string cyan("\033[0;36m");
    const std::string cyanBold("\033[1;36m");
    const std::string magenta("\033[0;35m");
    const std::string magentaBold("\033[1;35m");
    const std::string reset("\033[0m");
} // namespace COLORS

template<typename ...Args>
inline std::string sjoin(Args&&... args) noexcept
{
    std::ostringstream msg;
    //  ((msg << std::forward<Args>(args) << " "), ...);
    (msg << ... << args);
    return msg.str();
}


namespace format_helper
{

    template <class Src>
    inline Src cast(Src v)
    {
        return v;
    }

    inline const char *cast(const std::string& v)
    {
        return v.c_str();
    }
};

// taken from https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf/15341216
template <typename... Ts>
inline std::string sformat(const std::string &fmt, Ts&&... vs)
{
    using namespace format_helper;
    char b;
    
    //not counting the terminating null character.
    size_t required = std::snprintf(&b, 0, fmt.c_str(), cast(std::forward<Ts>(vs))...);
    std::string result;
    //because we use string as container, it adds extra 0 automatically
    result.resize(required , 0);
    //and snprintf will use n-1 bytes supplied
    std::snprintf(const_cast<char*>(result.data()), required + 1, fmt.c_str(), cast(std::forward<Ts>(vs))...);

    return result;
}

inline std::string timeStamp() {
    std::ostringstream strStream;
    std::time_t t = std::time(nullptr);
    strStream << "[" << std::put_time(std::localtime(&t), "%F %T") << "] ";
    return strStream.str();
}

#endif //STRINGFUNCTIONS_H
