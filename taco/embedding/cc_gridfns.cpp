#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <math.h>

namespace py = pybind11;


/*
 * \brief Calculate the Coulomb repulsion potential between two densities.
 *
 * \param grid0 Grid where first density and output is evaluated.
 * \param grid1 Integration grid where second density is evaluated.
 * \param weigths1 Integration weights for grid1.
 * \param density1 Density evaluated at grid1.
 */
py::array_t<double> coulomb_potential_grid(py::array_t<double, py::array::c_style> grid0,
                       py::array_t<double, py::array::c_style> grid1,
                       py::array_t<double, py::array::c_style> weights1,
                       py::array_t<double, py::array::c_style> density1){
    // Get the information from the python arrays
    py::buffer_info buf0 = grid0.request(), buf1 = grid1.request();
    py::buffer_info bufdens1 = density1.request();
    py::buffer_info bufw = weights1.request();
    // Make output array
    auto output = py::array_t<double>(buf0.shape[0]);
    py::buffer_info bufout = output.request();

    // now make cpp arrays
    double *cgrid0 = (double *) buf0.ptr,
           *cgrid1 = (double *) buf1.ptr,
           *cdens1 = (double *) bufdens1.ptr,
           *cweights1 = (double *) bufw.ptr,
           *coutput = (double *) bufout.ptr;

    for(int i=0; i<buf0.shape[0]; i++){
        // Loop over grid1 and integrate
        for(int j=0; j<buf1.shape[0]; j++){
            std::array<double, 3> d;
            // Check something
            d[0] = pow(cgrid0[i*3] - cgrid1[j*3], 2);
            d[1] = pow(cgrid0[i*3+1] - cgrid1[j*3+1], 2);
            d[2] = pow(cgrid0[i*3+2] - cgrid1[j*3+2], 2);
            double distance = sqrt(d[0] + d[1] + d[2]);
            if (distance > 1e-5){ // avoid very short distances
                coutput[i] += cweights1[j]*cdens1[j]/distance;
            }
        }
    }
    return output;
}


/*
 * \brief Calculate the Nuclear-electron attraction energy.
 *
 * \param charges Nuclear/effective charges
 * \param coords Coordinates of the nuclear charges
 * \param grid Integration grid where the density is evaluated.
 * \param weigths Integration weights for grid.
 * \param density Density evaluated at grid.
 */
double nuclear_attraction_energy(py::array_t<double, py::array::c_style> charges,
                       py::array_t<double, py::array::c_style> coords,
                       py::array_t<double, py::array::c_style> grid,
                       py::array_t<double, py::array::c_style> weights,
                       py::array_t<double, py::array::c_style> density){
    // Get the information from the python arrays
    py::buffer_info buf0 = coords.request(), buf1 = grid.request();
    py::buffer_info bufchar = charges.request();
    py::buffer_info bufdens = density.request();
    py::buffer_info bufw = weights.request();
    double natt_energy = 0.0;

    // now make cpp arrays
    double *ccoords = (double *) buf0.ptr,
           *cchar = (double *) bufchar.ptr,
           *cgrid = (double *) buf1.ptr,
           *cdens = (double *) bufdens.ptr,
           *cweights = (double *) bufw.ptr;

    for(int i=0; i<buf0.shape[0]; i++){
        // Loop over grid1 and integrate
        for(int j=0; j<buf1.shape[0]; j++){
            std::array<double, 3> d;
            // Check something
            d[0] = pow(ccoords[i*3] - cgrid[j*3], 2);
            d[1] = pow(ccoords[i*3+1] - cgrid[j*3+1], 2);
            d[2] = pow(ccoords[i*3+2] - cgrid[j*3+2], 2);
            double distance = sqrt(d[0] + d[1] + d[2]);
            if (distance > 1e-5){ // avoid very short distances
                natt_energy += cweights[j]*cchar[i]*cdens[j]/distance;
            }
        }
    }
    return natt_energy;
}


/*
 *   \brief Integrate something on grid.
 *
 *   \param  weights
 *   \params values
 */
double integrate(py::array_t<double, py::array::c_style> weights,
                 py::array_t<double, py::array::c_style> values){

    // Get the information from the python arrays
    py::buffer_info buf0 = weights.request(), buf1 = values.request();

    // Both arrays should be of the same size
    if (buf0.size != buf1.size){
        throw std::runtime_error("Size of arrays must be the same");
    }

    double *cweights = (double *) buf0.ptr,
           *cvalues = (double *) buf1.ptr;

    double result = 0.0;
    for (ssize_t i=0; i<buf0.size; i++){
        result += cweights[i]*cvalues[i];
    }
    return result;
};


PYBIND11_MODULE(cc_gridfns, m){
    m.doc() = "Evaluate stuff on grids.";
    m.def("coulomb_potential_grid", &coulomb_potential_grid,
          "The coulomb repulsion between two electronic densities.");
    m.def("nuclear_attraction_energy", &nuclear_attraction_energy,
          "The coulomb repulsion between two electronic densities.");
    m.def("integrate", &integrate,
          "Numerical integration. Sum over values and multiply by weights.");
}
