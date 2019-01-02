#include <pybind11/pybind11.h>
#include <hfts_grasp_planner/placement/reachability/reachability.h>

namespace py = pybind11;
using np_array = py::array_t<double, py::array::c_style | py::array::forcecast>;

PYBIND11_PLUGIN(yumi_reachability) {
    py::module m("yumi_reachability", "Reachability map for the Yumi robot.");
    py::class_<YumiReachabilityMap>(m, "YumiReachabilityMap")
        .def(py::init<np_array&>())
        .def("query", &YumiReachabilityMap::query);
    // m.def("hello_world", &hello_world, "A function that says hello");
    return m.ptr();
}
