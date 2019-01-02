#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <boost/heap/binomial_heap.hpp>

namespace py = pybind11;
using np_d_array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using np_b_array = py::array_t<bool, py::array::c_style | py::array::forcecast>;

struct RelativeIndex {
    int x, y, z;
    RelativeIndex(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
    RelativeIndex() : x(0), y(0), z(0) {}
};

struct Index {
    unsigned int x, y, z;

    Index(unsigned int x_, unsigned int y_, unsigned int z_) {
        x = x_;
        y = y_;
        z = z_;
    }

    Index() {
        x = 0;
        y = 0;
        z = 0;
    }

    void set(unsigned int x_, unsigned int y_, unsigned int z_) {
        x = x_;
        y = y_;
        z = z_;
    }

    // Return tuple (valid, index) where index is the neighboring index of this one, given the relative
    // index rel_index. The boolean valid is true if this index is within bounds, else false.
    std::tuple<bool, Index> getNeighbor(const RelativeIndex& rel_index, const size_t* bounds) const {
        int nx = x + rel_index.x;
        int ny = y + rel_index.y;
        int nz = z + rel_index.z;
        if (nx < 0 || nx >= bounds[0] ||
            ny < 0 || ny >= bounds[1] ||
            nz < 0 || nz >= bounds[2]) {
                return std::make_tuple(false, Index());
        }
        return std::make_tuple(true, Index(nx, ny, nz));
    }
};

struct PQElement {
    unsigned int id;
    double key;

    PQElement() : id(0), key(0) {}

    PQElement(unsigned int id_, double key_) {
        id = id_;
        key = key_;
    }

    // PQElement& operator=(const PQElement& other) = default;
};

struct PQElementComparer {
    constexpr bool operator()(const PQElement& a, const PQElement& b) const {
        return a.key > b.key;
    }
};

struct CellInfo {
    double distance; // distance to closest background cell
    unsigned int id; // 1d index of this cell
    Index cell_idx; // index of this cell
    Index closest_bg_idx; // index of closest background (0) cell
    boost::heap::binomial_heap<PQElement, boost::heap::compare<PQElementComparer> >::handle_type pq_handle;
    bool in_pq;
    CellInfo() : in_pq(false), distance(std::numeric_limits<double>::max()) {}
};

double compute_distance(const Index& a, const Index& b) {
    int dx = (int)a.x - (int)b.x;
    int dy = (int)a.y - (int)b.y;
    int dz = (int)a.z - (int)b.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

void computeMinSpanningtree(const np_b_array& input_arr, const std::vector<RelativeIndex>& neighbors, np_d_array& out_arr) {
    std::vector<CellInfo> cells;  // stores information for each cell
    cells.reserve(input_arr.size());
    // priority queue of cells sorted by distance to background pixel
    boost::heap::binomial_heap<PQElement, boost::heap::compare<PQElementComparer> > pq;
    // init cells and pq, run over the full input array
    for (size_t i = 0; i < input_arr.shape(0); ++i) {
        for (size_t j = 0; j < input_arr.shape(1); ++j) {
            for (size_t k = 0; k < input_arr.shape(2); ++k) {
                CellInfo new_item;
                new_item.distance = std::numeric_limits<double>::infinity();
                new_item.cell_idx.set(i, j, k);
                new_item.closest_bg_idx.set(i, j, k);
                new_item.id = cells.size();
                // if this voxel is 0, it is a start state
                if (!*input_arr.data(i, j, k)) {
                    new_item.pq_handle = pq.emplace(PQElement(new_item.id, 0.0));
                    new_item.in_pq = true;
                    new_item.distance = 0.0;
                }
                cells.push_back(new_item);
            }
        }
    }
    // now compute distance field (essentially running dijkstra)
    while (!pq.empty()) {
        // std::cout << "PQ size: " << pq.size() << std::endl;
        // get the cell with minimal distance
        PQElement curr_elem = pq.top();
        pq.pop();
        const CellInfo& current_cell = cells[curr_elem.id];
        // std::cout << "Current element " << curr_elem.id << " has key " << curr_elem.key << std::endl;
        // std::cout << "Cell distance: " << current_cell.distance << std::endl;
        assert(curr_elem.key == current_cell.distance);
        // run over neighbors of this cell
        for (auto& neigh : neighbors) {
            bool bvalid = false;
            Index neigh_index;
            std::tie(bvalid, neigh_index) = current_cell.cell_idx.getNeighbor(neigh, input_arr.shape());
            // if this neighbor is within bounds
            if (bvalid) {
                // compute distance
                double new_distance = compute_distance(neigh_index, current_cell.closest_bg_idx);
                // compute 1d index
                unsigned int nid = input_arr.index_at(neigh_index.x, neigh_index.y, neigh_index.z);
                // if this distance is better than current distance
                if (new_distance < cells[nid].distance) {
                    cells[nid].distance = new_distance;
                    cells[nid].closest_bg_idx = current_cell.closest_bg_idx;
                    if (cells[nid].in_pq) {
                        pq.decrease(cells[nid].pq_handle, PQElement(nid, new_distance));
                    } else {
                        cells[nid].pq_handle = pq.push(PQElement(nid, new_distance));
                        cells[nid].in_pq = true;
                    }
                }
            }
        }
    }
    // copy distances into out_arr
    double* out_data = out_arr.mutable_data();
    for (auto& cell_info : cells) {
        out_data[cell_info.id] = cell_info.distance;
    }
}

void computeDF(const np_b_array& input_arr, const np_b_array& adj_arr, np_d_array& out_arr) {
    if (input_arr.ndim() != out_arr.ndim()) {
        throw std::runtime_error("Input and output array must have same dimension.");
    }
    if (input_arr.size() != out_arr.size()) {
        throw std::runtime_error("Input and output array must have same size");
    }
    if (input_arr.ndim() != 3) {
        throw std::runtime_error("Input array must have dimension 3");
    }
    if (!out_arr.writeable()) {
        throw std::runtime_error("Output array must be writable");
    }
    if (adj_arr.ndim() != 3 or adj_arr.shape(0) != 3 or adj_arr.shape(1) != 3 or adj_arr.shape(2) !=3) {
        throw std::runtime_error("Adjacency array must have shape (3, 3, 3).");
    }
    // construct a vector of relative neighbor indices
    std::vector<RelativeIndex> neighbors;
    for (size_t i = 0; i < adj_arr.shape(0); ++i) {
        for (size_t j = 0; j < adj_arr.shape(1); ++j) {
            const bool* adj_arr_row = adj_arr.data(i, j);
            for (size_t k = 0; k < adj_arr.shape(2); ++k) {
                if (adj_arr_row[k] and !(i == 1 and j == 1 and k == 1)) {
                    neighbors.emplace_back(RelativeIndex(i - 1, j - 1, k - 1));
                }
            }
        }
    }
    // std::cout << "Neighbors: " << neighbors.size() << std::endl;
    computeMinSpanningtree(input_arr, neighbors, out_arr);
}

PYBIND11_PLUGIN(clearance_utils) {
    py::module m("clearance_utils", "Utility function to compute a clearance map");
    m.def("compute_df", &computeDF);
    // m.def("hello_world", &hello_world, "A function that says hello");
    return m.ptr();
}