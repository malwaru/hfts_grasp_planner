#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <ompl/datastructures/NearestNeighborsGNATNoThreadSafety.h>

struct EndEffectorPose {
    double pos[3];
    double quat[4];
    unsigned int id;
    EndEffectorPose(double x, double y, double z, double w, double i, double j, double k, unsigned int id_) {
        pos[0] = x;
        pos[1] = y;
        pos[2] = z;
        quat[0] = w;
        quat[1] = i;
        quat[2] = j;
        quat[3] = k;
        id = id_;
    }

    EndEffectorPose() {
        pos[0] = 0.0;
        pos[1] = 0.0;
        pos[2] = 0.0;
        quat[0] = 1.0;
        quat[1] = 0.0;
        quat[2] = 0.0;
        quat[3] = 0.0;
        id = 0;
    }

    bool operator==(const EndEffectorPose& other) const {
        return pos[0] == other.pos[0] and pos[1] == other.pos[1] and pos[2] == other.pos[2]
            and quat[0] == other.quat[0] and quat[1] == other.quat[1] and quat[2] == other.quat[2]
            and quat[3] == other.quat[3];
    }

    bool operator!=(const EndEffectorPose& other) const {
        return !((*this) == other);
    }
};

class YumiReachabilityMap {
    public:
        YumiReachabilityMap(pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>& data);
        ~YumiReachabilityMap();

        pybind11::tuple query(pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>& query_point);

    private:
        ompl::NearestNeighborsGNATNoThreadSafety<EndEffectorPose> _gnat;
};