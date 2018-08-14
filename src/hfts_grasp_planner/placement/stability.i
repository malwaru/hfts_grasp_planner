%module placement_stability
%{
    #include <boost/python.hpp>
    void test_env(boost::python::api::object *object);
    void count(unsigned int n);
%}
void test_env(boost::python::api::object *object);
void count(unsigned int n);