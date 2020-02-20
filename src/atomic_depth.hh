/*////////////////////////////////////////////////////////////////
Permission to use, copy, modify, and distribute this program for
any purpose, with or without fee, is hereby granted, provided that
the notices on the head, the reference information, and this
copyright notice appear in all copies or substantial portions of
the Software. It is provided "as is" without express or implied
warranty.
*////////////////////////////////////////////////////////////////

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;


const static char nb[26][3]={{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1},
{1,1,0}, {1,-1,0}, {-1,1,0}, {-1,-1,0}, {1,0,1}, {1,0,-1}, {-1,0,1}, {-1,0,-1}, {0,1,1}, {0,1,-1}, {0,-1,1}, {0,-1,-1},
{1,1,1}, {1,1,-1}, {1,-1,1}, {-1,1,1}, {1,-1,-1}, {-1,-1,1}, {-1,1,-1}, {-1,-1,-1}};

const int fast_one_shell_cut[3] = {6, 12, 9};
const int fast_one_shell_lowb[3] = {0, 6, 18};
const int fast_one_shell_highb[3] = {6, 18, 26};

typedef struct point3d
{
	double x,y,z;
}point3d;
typedef struct volumepixel
{
	float distance;
	bool inout;
	bool isbound;
	bool isdone;
}volumepixel;
typedef struct voxel
{
	int ix,iy,iz;
}voxel;
typedef struct voxel2
{
	short int ix,iy,iz;
}voxel2;
typedef struct faceinfo
{
	int a,b,c;
	point3d pn;
	double area;
	bool inout;//interior true
}faceinfo;
typedef struct vertinfo
{
	double x,y,z;
	point3d pn;
	double area;
	int atomid;
	bool inout,iscont;//is concave surface
}vertinfo;

class AtomicDepth
{
public:
	AtomicDepth( py::array_t<double> & points, py::array_t<double> & radii,
		double probe_radius, double resolution, bool report_sasa , size_t smooth_iter );

	py::array_t<double> calcdepth( py::array_t<double> & points, py::array_t<double> & radii ) const;

	double inner_calcdepth( point3d const & point, double const & radius ) const;

	py::array_t<double> get_surface_vertex_bases();
	py::array_t<double> get_surface_vertex_normals();
	py::array_t<size_t> get_surface_face_vertices();
	py::array_t<double> get_surface_face_centers();
	py::array_t<double> get_surface_face_normals();
	py::array_t<double> get_surface_face_areas();

	void visualize_at_depth( double depth, std::string const & fname, double fraction ) const;

private:
	void boundbox(std::vector<point3d> const & points, std::vector<double> const & radii,point3d & minp,point3d & maxp);
	void boundingatom( std::vector<point3d> const & points, std::vector<double> const & radii );
	void initpara( std::vector<point3d> const & points, std::vector<double> const & radii );
	void fill_vp();
	void fillvoxels( std::vector<point3d> const & points, std::vector<double> const & radii );
	void fillatom(point3d const & point, double radius);
	void fastoneshell(int innum,int & allocout,std::vector<std::vector<std::vector<voxel2> > > & boundpoint,int & outnum, int & elimi);
	void fastdistancemap(int type);
	void buildboundary();

	void checkEuler();
	void computenorm();
	void marchingcubeinit(int stype );
	void marchingcube(int stype );
	void scale_surface();
	void laplaciansmooth(int numiter);

	size_t get_atom_type( double radius ) { return int( radius * 100 ); }
	double atom_type_to_radius( size_t type ) { return double(type) / 100.0; }

private:
	point3d ptran_;
	int boxlength_;
	bool flagradius_;
	double proberadius_;
	double fixsf_;
	double scalefactor_;
	point3d pmin_,pmax_;
	int pheight_,pwidth_,plength_;
	std::vector<int> widxz_;
	std::vector<std::vector<int> > depty_;
	double cutradis_;
	std::vector<std::vector<std::vector<volumepixel > > > vp_;
	std::shared_ptr<std::vector<voxel2> > inarray_;
	std::shared_ptr<std::vector<voxel2> > outarray_;
	int totalsurfacevox_;
	int totalinnervox_;

	std::vector<vertinfo> verts_;
	std::vector<faceinfo> faces_;

};


void declare_AtomicDepth(const py::module& m, const std::string& class_name) {

  py::class_<AtomicDepth>(m, class_name.c_str())
      .def(py::init<py::array_t<double> &, py::array_t<double> &, double, double, bool, size_t>())

      .def("calcdepth", &AtomicDepth::calcdepth)
      .def("visualize_at_depth", &AtomicDepth::visualize_at_depth)
      .def("get_surface_vertex_bases", &AtomicDepth::get_surface_vertex_bases)
      .def("get_surface_vertex_normals", &AtomicDepth::get_surface_vertex_normals)
      .def("get_surface_face_vertices", &AtomicDepth::get_surface_face_vertices)
      .def("get_surface_face_centers", &AtomicDepth::get_surface_face_centers)
      .def("get_surface_face_normals", &AtomicDepth::get_surface_face_normals)
      .def("get_surface_face_areas", &AtomicDepth::get_surface_face_areas);


}

PYBIND11_MODULE(_atomic_depth, m) {
    m.doc() = "AtomicDepth yo";
    declare_AtomicDepth(m, "AtomicDepth");

}

