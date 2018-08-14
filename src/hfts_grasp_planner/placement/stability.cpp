#include <iostream>
// #include <openrave-0.9/python/binding.h>
#include <openrave-0.9/python/openravepy_int.h>
#include <boost/python.hpp>

void test_env(boost::python::api::object *object)
{
    // TODO probably a better idea to use boost python for this than swig
    std::cout << "called test_env(...)" << std::endl;
}

void count(unsigned int n)
{
    for (unsigned int i = 0; i < n; ++i)
    {
        std::cout << "Counting " << i << std::endl;
    }
}