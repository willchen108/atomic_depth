/*////////////////////////////////////////////////////////////////
Permission to use, copy, modify, and distribute this program for
any purpose, with or without fee, is hereby granted, provided that
the notices on the head, the reference information, and this
copyright notice appear in all copies or substantial portions of
the Software. It is provided "as is" without express or implied
warranty.
*////////////////////////////////////////////////////////////////


#include <atomic_depth.hh>
#include <fstream>
#include <stdio.h>


void
my_assert(bool value, std::string const & message="Error!") {
	if ( ! value ) {
		throw std::runtime_error("Error!");
	}
}

std::vector<point3d>
numpy_to_cpp_xyz( py::array_t<double> & in_points ) {
	py::buffer_info points_buffer = in_points.request();
    double * points_array_ptr = (double *) points_buffer.ptr;
    my_assert( points_buffer.shape[0] % 3 == 0, "Points not divisible by 3" );
    std::vector<point3d> points( points_buffer.shape[0] / 3 );
    for ( size_t i = 0; i < points.size(); i++ ) {
    	points[i].x = points_array_ptr[i*3+0];
    	points[i].y = points_array_ptr[i*3+1];
    	points[i].z = points_array_ptr[i*3+2];
    }
    return points;
}

std::vector<double>
numpy_to_cpp( py::array_t<double> & in_radii ) {
    py::buffer_info radii_buffer = in_radii.request();
    double * radii_array_ptr = (double *) radii_buffer.ptr;
    std::vector<double> radii( radii_buffer.shape[0] );
    for ( size_t i = 0; i < radii.size(); i++ ) radii[i] = radii_array_ptr[i];

    return radii;
}

py::array_t<double>
cpp_to_numpy( std::vector<double> const & points ) {

    py::array_t<double> result_array = py::array_t<double> ( points.size() );
    py::buffer_info result_array_buffer = result_array.request();
    double * result_array_ptr = (double *) result_array_buffer.ptr;

    for ( size_t i = 0; i < points.size(); i++ ) {
    	result_array_ptr[i] = points[i];
    }

    return result_array;
}

py::array_t<size_t>
cpp_to_numpy( std::vector<size_t> const & points ) {

    py::array_t<size_t> result_array = py::array_t<size_t> ( points.size() );
    py::buffer_info result_array_buffer = result_array.request();
    size_t * result_array_ptr = (size_t *) result_array_buffer.ptr;

    for ( size_t i = 0; i < points.size(); i++ ) {
    	result_array_ptr[i] = points[i];
    }

    return result_array;
}

py::array_t<double>
cpp_xyz_to_numpy( std::vector<point3d> const & points ) {

    py::array_t<double> result_array = py::array_t<double> ( points.size()*3 );
    py::buffer_info result_array_buffer = result_array.request();
    double * result_array_ptr = (double *) result_array_buffer.ptr;

    for ( size_t i = 0; i < points.size(); i++ ) {
    	result_array_ptr[i*3+0] = points[i].x;
    	result_array_ptr[i*3+1] = points[i].y;
    	result_array_ptr[i*3+2] = points[i].z;
    }

    return result_array;
}



AtomicDepth::AtomicDepth( py::array_t<double> & in_points, py::array_t<double> & in_radii,
		double probe_radius, double resolution, bool report_sasa , size_t smooth_iter   )
{
	std::vector<point3d> points = numpy_to_cpp_xyz( in_points );
	std::vector<double> radii = numpy_to_cpp( in_radii );

	// std::cout << "Points: " << points.size() << " Radii: " << radii.size() << std::endl;

	my_assert( points.size() == radii.size(), "Num points / 3 not equal to num radii" );


	boxlength_=128;
	flagradius_=false;
	scalefactor_=1;
	proberadius_=probe_radius;
	widxz_.resize(13);
	depty_.resize(13);
	// for(i=0;i<13;i++)
	//  depty_[i]=NULL;
	vp_.clear();
	pheight_=0;
	pwidth_=0;
	plength_=0;

	// This code originally specified a scale factor instead of a resolution.
	// After backtracing the calculation, the scale factor for a given resolution
	// can be determined from the formula below.
	fixsf_=double(1.0)/resolution * double(127.0f)/double(128.0f);

	initpara( points, radii );
	// std::cout << boost::str(boost::format("actual boxlength %3d, box[%3d*%3d*%3d], resolution %6.3f\n")%
	// 	boxlength_%plength_%pwidth_%pheight_%(1.0f/scalefactor_)) << std::endl;
	fillvoxels( points, radii );
	buildboundary();
	if ( ! report_sasa ) {
		fastdistancemap( 1 );
	} else {
		fastdistancemap( 1 );
		marchingcube(4);
		checkEuler();
		computenorm();

		if ( smooth_iter > 0 ) {
			laplaciansmooth( smooth_iter );
			computenorm();
		}

		scale_surface();
		computenorm();
	}
}

py::array_t<double> AtomicDepth::calcdepth( py::array_t<double> & in_points, py::array_t<double> & in_radii ) const
{
	std::vector<point3d> points = numpy_to_cpp_xyz( in_points );
	std::vector<double> radii = numpy_to_cpp( in_radii );

	my_assert( points.size() == radii.size(), "Num points / 3 not equal to num radii" );

	std::vector<double> depval( points.size() );

	for ( size_t i = 0; i < points.size(); i++ ) {
		depval[i] = inner_calcdepth( points[i], radii[i] );
	}

	return cpp_to_numpy( depval );;
}

double AtomicDepth::inner_calcdepth( point3d const & point, double const & radius ) const
{
	int ox,oy,oz;
	point3d cp;
	double depth;

	cp.x=point.x+ptran_.x;
	cp.y=point.y+ptran_.y;
	cp.z=point.z+ptran_.z;
	cp.x*=scalefactor_;
	cp.y*=scalefactor_;
	cp.z*=scalefactor_;
	ox=int(cp.x+0.5);
	oy=int(cp.y+0.5);
	oz=int(cp.z+0.5);

	if ( ox >= 0 && oy >= 0 && oz >= 0 && ox < plength_ && oy < pwidth_ && oz < pheight_ ) {
		depth = vp_[ox][oy][oz].distance/scalefactor_-proberadius_;
		if ( depth < radius ) depth=radius;
	} else {
		depth = radius;
	}

	return depth;
}

void AtomicDepth::visualize_at_depth( double depth, std::string const & fname, double fraction ) const {
	int anum=1, rnum=1, i, j, k;

	std::ofstream out( fname );

	size_t dump_every = size_t( 1 / fraction );
	size_t count = 0;

	for ( i=0; i<plength_; i++ ) {
		for ( j=0; j<pwidth_; j++ ) {
			for ( k=0; k<pheight_; k++ ) {
				double this_depth = vp_[i][j][k].distance/scalefactor_-proberadius_;
				if ( this_depth < depth ) continue;
				count += 1;
				if ( count % dump_every != 0 ) continue;

				double x = i / scalefactor_ - ptran_.x;
				double y = j / scalefactor_ - ptran_.y;
				double z = k / scalefactor_ - ptran_.z;

				char buf[128];
				snprintf(buf,128,"%s%5i %4s %3s %c%4i    %8.3f%8.3f%8.3f%6.2f%6.2f %11s\n",
					"HETATM",
					anum++,
					"BURR",
					"BUR",
					'B',
					rnum++,
					x,y,z,
					1.0,
					1.0,
					"B"
				);
				out << buf;

				rnum %= 10000;
				anum %= 100000;
			}
		}
	}

	out.close();
}


py::array_t<double> 
AtomicDepth::get_surface_vertex_bases() {
	std::vector<point3d> points( verts_.size() );
	for ( size_t i = 0; i < verts_.size(); i++ ) {
		points[i].x = verts_[i].x;
		points[i].y = verts_[i].y;
		points[i].z = verts_[i].z;
	}
	return cpp_xyz_to_numpy(points); 
}

py::array_t<double> 
AtomicDepth::get_surface_vertex_normals() {
	std::vector<point3d> points( verts_.size() );
	for ( size_t i = 0; i < verts_.size(); i++ ) {
		points[i].x = verts_[i].pn.x;
		points[i].y = verts_[i].pn.y;
		points[i].z = verts_[i].pn.z;
	}
	return cpp_xyz_to_numpy(points); 
}

py::array_t<size_t> 
AtomicDepth::get_surface_face_vertices() {
	std::vector<size_t> points( faces_.size()*3 );
	for ( size_t i = 0; i < faces_.size(); i++ ) {
		points[i*3+0] = faces_[i].a;
		points[i*3+1] = faces_[i].b;
		points[i*3+2] = faces_[i].c;
	}
	return cpp_to_numpy(points); 
}

py::array_t<double> 
AtomicDepth::get_surface_face_centers() {
	std::vector<point3d> points( faces_.size() );
	for ( size_t i = 0; i < faces_.size(); i++ ) {
		vertinfo const & a = verts_[faces_[i].a];
		vertinfo const & b = verts_[faces_[i].b];
		vertinfo const & c = verts_[faces_[i].c];
		points[i].x = (a.x + b.x + c.x)/3;
		points[i].y = (a.y + b.y + c.y)/3;
		points[i].z = (a.z + b.z + c.z)/3;
	}
	return cpp_xyz_to_numpy(points); 
}

py::array_t<double> 
AtomicDepth::get_surface_face_normals() {
	std::vector<point3d> points( faces_.size() );
	for ( size_t i = 0; i < faces_.size(); i++ ) {
		points[i].x = faces_[i].pn.x;
		points[i].y = faces_[i].pn.y;
		points[i].z = faces_[i].pn.z;
	}
	return cpp_xyz_to_numpy(points); 
}

py::array_t<double> 
AtomicDepth::get_surface_face_areas() {
	std::vector<double> points( faces_.size() );
	for ( size_t i = 0; i < faces_.size(); i++ ) {
		points[i] = faces_[i].area;
	}
	return cpp_to_numpy(points); 
}



void AtomicDepth::boundbox( std::vector<point3d> const & points, std::vector<double> const &,
	point3d & minp,point3d & maxp)
{
	minp.x=100000;minp.y=100000;minp.z=100000;
	maxp.x=-100000;maxp.y=-100000;maxp.z=-100000;

	for ( size_t i = 0; i < points.size(); i++ ) {
		point3d const & xyz = points[i];
		if ( xyz.x<minp.x ) {
			minp.x=xyz.x;
		}
		if ( xyz.y<minp.y ) {
			minp.y=xyz.y;
		}
		if ( xyz.z<minp.z ) {
			minp.z=xyz.z;
		}
		if ( xyz.x>maxp.x ) {
			maxp.x=xyz.x;
		}
		if ( xyz.y>maxp.y ) {
			maxp.y=xyz.y;
		}
		if ( xyz.z>maxp.z ) {
			maxp.z=xyz.z;
		}
	}
}
void AtomicDepth::buildboundary()
{
	int i,j,k;
	int ii;
	bool flagbound;
	for ( i=0; i<plength_; i++ ) {
		for ( j=0; j<pheight_; j++ ) {
			for ( k=0; k<pwidth_; k++ ) {
				if ( vp_[i][k][j].inout ) {
					//6 neighbors
					//                  if(( k-1>-1 && !vp_[i][k-1][j].inout) || ( k+1<pwidth_ &&!vp_[i][k+1][j].inout)
					//                  || ( j-1>-1 && !vp_[i][k][j-1].inout) || ( j+1<pheight_ &&!vp_[i][k][j+1].inout)
					//                  || ( i-1>-1 && !vp_[i-1][k][j].inout) || ( i+1<plength_ &&!vp_[i+1][k][j].inout))
					//                      vp_[i][k][j].isbound=true;
					//  /*
					//26 neighbors
					flagbound=false;
					ii=0;
					while ( !flagbound && ii<26 )
							{
						if ( i+nb[ii][0]>-1 && i+nb[ii][0]<plength_
								&& k+nb[ii][1]>-1 && k+nb[ii][1]<pwidth_
								&& j+nb[ii][2]>-1 && j+nb[ii][2]<pheight_
								&& !vp_[i+nb[ii][0]][k+nb[ii][1]][j+nb[ii][2]].inout ) {
							vp_[i][k][j].isbound=true;
							flagbound=true;
						} else ii++;
					}
					//      */
				}
			}

		}
	}
}

void AtomicDepth::boundingatom( std::vector<point3d> const & points, std::vector<double> const & radii )
{

	double max_radius = 0;
	for ( size_t i = 0; i < radii.size(); i++ ) {
		max_radius = std::max<double>( max_radius, radii[i] );
	}

	// std::vector<bool> atom_list( get_atom_type( max_radius) + 1, false );

	// for ( size_t i = 0; i < radii.size(); i++ ) {
	// 	atom_list[ get_atom_type( radii[i] ) ] = true;
	// }

	size_t num_types = get_atom_type( max_radius ) + 1;


	widxz_.clear();
	depty_.clear();

	widxz_.resize( num_types );
	depty_.resize( num_types );


	for ( size_t atype = 0; atype < num_types; atype++ ) {
		depty_[atype].clear();

		double tradius = (atom_type_to_radius(atype) + proberadius_)*scalefactor_+0.5;
		// double tradius = (radii[ atype ] + proberadius_)*scalefactor_+0.5;
		double sradius = tradius*tradius;
		widxz_[atype] = int(tradius)+1;

		depty_[atype].resize(widxz_[atype]*widxz_[atype]);

		size_t indx = 0;
		for ( int j = 0; j < widxz_[atype]; j++ ) {
			for ( int k = 0; k < widxz_[atype]; k++ ) {
				double txz = j*j+k*k;
				if ( txz > sradius ) {
					depty_[atype][indx]=-1;
				} else {
					double tdept = sqrt(sradius-txz);
					depty_[atype][indx] = int(tdept);
				}
				indx++;
			}
		}
	}

}


void AtomicDepth::initpara( std::vector<point3d> const & points, std::vector<double> const & radii )
{
	double fmargin=2.5;
	vp_.clear();
	boundbox(points, radii,pmin_,pmax_);


	pmin_.x-=proberadius_+fmargin;
	pmin_.y-=proberadius_+fmargin;
	pmin_.z-=proberadius_+fmargin;
	pmax_.x+=proberadius_+fmargin;
	pmax_.y+=proberadius_+fmargin;
	pmax_.z+=proberadius_+fmargin;


	ptran_.x=-pmin_.x;
	ptran_.y=-pmin_.y;
	ptran_.z=-pmin_.z;
	scalefactor_=pmax_.x-pmin_.x;
	if ( (pmax_.y-pmin_.y)>scalefactor_ ) {
		scalefactor_=pmax_.y-pmin_.y;
	}
	if ( (pmax_.z-pmin_.z)>scalefactor_ ) {
		scalefactor_=pmax_.z-pmin_.z;
	}
	scalefactor_=(boxlength_-1.0)/double(scalefactor_);
	///////////////////////////add this automatically first fix sf then fix boxlength_
	//  /*
	boxlength_=int(boxlength_*fixsf_/scalefactor_);
	scalefactor_=fixsf_;
	double threshbox=300;
	if ( boxlength_>threshbox ) {
		double sfthresh=threshbox/double(boxlength_);
		boxlength_=int(threshbox);
		scalefactor_=scalefactor_*sfthresh;
	}
	//  */

	plength_=int(ceil(scalefactor_*(pmax_.x-pmin_.x))+1);
	pwidth_=int(ceil(scalefactor_*(pmax_.y-pmin_.y))+1);
	pheight_=int(ceil(scalefactor_*(pmax_.z-pmin_.z))+1);
	if ( plength_>boxlength_ ) {
		plength_=boxlength_;
	}
	if ( pwidth_>boxlength_ ) {
		pwidth_=boxlength_;
	}
	if ( pheight_>boxlength_ ) {
		pheight_=boxlength_;
	}

	boundingatom( points, radii );
	cutradis_=proberadius_*scalefactor_;
}
// Fill the 9 voxels centered at this xyz with .inout = true
void AtomicDepth::fillatom( point3d const & point, double radius )
{
	int cx,cy,cz;
	point3d cp;
	cp.x=point.x+ptran_.x;
	cp.y=point.y+ptran_.y;
	cp.z=point.z+ptran_.z;
	cp.x*=scalefactor_;
	cp.y*=scalefactor_;
	cp.z*=scalefactor_;
	cx=int(cp.x+0.5);
	cy=int(cp.y+0.5);
	cz=int(cp.z+0.5);
	int at=get_atom_type(radius);
	int i,j,k;
	int ii,jj,kk;
	int mi,mj,mk;
	int si,sj,sk;
	int nind=0;
	for ( i=0; i<widxz_[at]; i++ ) {
		for ( j=0; j<widxz_[at]; j++ ) {
			if ( depty_[at][nind]!=-1 ) {

				for ( ii=-1; ii<2; ii++ ) {
					for ( jj=-1; jj<2; jj++ ) {
						for ( kk=-1; kk<2; kk++ ) {
							if ( ii!=0 && jj!=0 && kk!=0 ) {
								mi=ii*i;
								mk=kk*j;
								for ( k=0; k<=depty_[at][nind]; k++ ) {
									mj=k*jj;
									si=cx+mi;
									sj=cy+mj;
									sk=cz+mk;
									if ( si<0 || sj<0 || sk<0 || si>=plength_ || sj>=pwidth_ || sk>=pheight_ ) {
										continue;
									}

									if ( vp_[si][sj][sk].inout==false ) {
										vp_[si][sj][sk].inout=true;
									}
								}//k

							}//if
						}//kk
					}//jj
				}//ii


			}//if
			nind++;
		}//j
	}//i
}
void AtomicDepth::fill_vp() {
	int i,j;
	vp_.resize( plength_ );
	for ( i=0; i<plength_; i++ ) {
		vp_[i].resize(pwidth_);
	}
	for ( i=0; i<plength_; i++ ) {
		for ( j=0; j<pwidth_; j++ ) {
			vp_[i][j].resize(pheight_);
		}
	}
}

// Prepare the voxel array used in the calculations with the pose
// all heavy atoms get placed at voxels and those locations get .inout = true and .isdone = true
void AtomicDepth::fillvoxels( std::vector<point3d> const & points, std::vector<double> const & radii )
{

	int i,j,k;
	if ( vp_.size() == 0 ) {
		fill_vp();
	}

	for ( i=0; i<plength_; i++ ) {
		for ( j=0; j<pwidth_; j++ ) {
			for ( k=0; k<pheight_; k++ ) {
				vp_[i][j][k].inout=false;
				vp_[i][j][k].isdone=false;
				vp_[i][j][k].isbound=false;
				vp_[i][j][k].distance=-1;
			}
		}
	}

	for ( size_t i = 0; i < points.size(); i++ ) {
		fillatom( points[i], radii[i] );
	}

	for ( i=0; i<plength_; i++ ) {
		for ( j=0; j<pwidth_; j++ ) {
			for ( k=0; k<pheight_; k++ ) {
				if ( vp_[i][j][k].inout ) {
					vp_[i][j][k].isdone=true;
				}
			}
		}
	}
}


// Flood fill near points marked as boundary
// If zero is passed as type, only flood fill enough to fill cutradis_ away from atomic centers
void AtomicDepth::fastdistancemap(int type)
{
	int i,j,k;
	int positin,positout,eliminate;
	totalsurfacevox_=0;
	totalinnervox_=0;

	std::vector<std::vector<std::vector<voxel2> > > boundpoint(plength_);
	for ( i=0; i<plength_; i++ ) {
		boundpoint[i].resize(pwidth_);
	}
	for ( i=0; i<plength_; i++ ) {
		for ( j=0; j<pwidth_; j++ ) {
			boundpoint[i][j].resize(pheight_);
		}
	}
	for ( i=0; i<plength_; i++ ) {
		for ( j=0; j<pwidth_; j++ ) {
			for ( k=0; k<pheight_; k++ ) {
				vp_[i][j][k].isdone=false;
				if ( vp_[i][j][k].inout ) {
					if ( vp_[i][j][k].isbound ) {
						totalsurfacevox_++;
						boundpoint[i][j][k].ix=i;
						boundpoint[i][j][k].iy=j;
						boundpoint[i][j][k].iz=k;
						vp_[i][j][k].distance=0;
						vp_[i][j][k].isdone=true;
					} else {
						totalinnervox_++;
					}
				}
			}
		}
	}
	int allocin=int(1.2*totalsurfacevox_);
	int allocout=int(1.2*totalsurfacevox_);
	if ( allocin>totalinnervox_ ) {
		allocin=totalinnervox_;
	}
	if ( allocin<totalsurfacevox_ ) {
		allocin=totalsurfacevox_;
	}
	if ( allocout>totalinnervox_ ) {
		allocout=totalinnervox_;
	}

	inarray_=std::make_shared<std::vector<voxel2> >( allocin );
	outarray_=std::make_shared<std::vector<voxel2> >( allocout );
	positin=0;positout=0;

	for ( i=0; i<plength_; i++ ) {
		for ( j=0; j<pwidth_; j++ ) {
			for ( k=0; k<pheight_; k++ ) {
				if ( vp_[i][j][k].isbound ) {
					(*inarray_)[positin].ix=i;
					(*inarray_)[positin].iy=j;
					(*inarray_)[positin].iz=k;
					positin++;
					vp_[i][j][k].isbound=false;//as flag of outarray_
				}
			}
		}
	}
	///////////////////////////////////////////////////
	if ( type==0 ) { //do part
		do {
			fastoneshell(positin, allocout, boundpoint, positout,eliminate);

			positin=0;
			for ( i=0; i<positout; i++ ) {
				vp_[(*outarray_)[i].ix][(*outarray_)[i].iy][(*outarray_)[i].iz].isbound=false;
				if ( vp_[(*outarray_)[i].ix][(*outarray_)[i].iy][(*outarray_)[i].iz].distance<=1.02*cutradis_ ) {
					(*inarray_)[positin].ix=(*outarray_)[i].ix;
					(*inarray_)[positin].iy=(*outarray_)[i].iy;
					(*inarray_)[positin].iz=(*outarray_)[i].iz;
					positin++;
				}
				if ( positin>=allocin ) {
					allocin*=2;
					if ( allocin>totalinnervox_ ) allocin=totalinnervox_;

					inarray_->resize(allocin);
				}
			}
		}
		while(positin!=0);
	} else if ( type==1 ) { //do all
		std::shared_ptr<std::vector<voxel2> > tpoint;
		do {

			fastoneshell( positin, allocout, boundpoint, positout,eliminate);//inarray_, outarray_,

			tpoint=inarray_;
			inarray_=outarray_;
			outarray_=tpoint;
			positin=positout;
			int alloctmp;
			alloctmp=allocin;
			allocin=allocout;
			allocout=alloctmp;
			for ( i=0; i<positin; i++ ) {
				vp_[(*inarray_)[i].ix][(*inarray_)[i].iy][(*inarray_)[i].iz].isbound=false;
			}


		}
		while(positout!=0);
	}

	inarray_ = nullptr;
	outarray_ = nullptr;

	double cutsf=scalefactor_-0.5;
	if ( cutsf<0 ) cutsf=0;
	//   cutsf=100000000;
	for ( i=0; i<plength_; i++ ) {
		for ( j=0; j<pwidth_; j++ ) {
			for ( k=0; k<pheight_; k++ ) {
				vp_[i][j][k].isbound=false;
				//ses solid
				if ( vp_[i][j][k].inout ) {
					if ( !vp_[i][j][k].isdone
							|| (vp_[i][j][k].isdone && vp_[i][j][k].distance>=cutradis_-0.50/(0.1+cutsf))//0.33  0.75/scalefactor_
							) {
						vp_[i][j][k].isbound=true;
					}
				}
			}
		}
	}
}

// Do one step of the flood-fill magic
void AtomicDepth::fastoneshell(int innum,int & allocout,std::vector<std::vector<std::vector<voxel2> > > & boundpoint, int & outnum, int & elimi)
{
	int i, number,positout;
	int tx,ty,tz;
	int dx,dy,dz;
	int eliminate=0;
	float squre;
	positout=0;
	number=innum;
	if ( number==0 ) return;
	//new code
	int j,q;
	voxel tnv;

	for ( q=0; q<3; q++ ) {

		for ( i=0; i<number; i++ ) {
			if ( positout>=allocout-fast_one_shell_cut[q] ) {
				allocout=int(1.2*allocout);
				if ( allocout>totalinnervox_ ) allocout=totalinnervox_;
				// outarray_=(voxel2 *)doubleloc(outarray_,(*allocout)*sizeof(voxel2));
				outarray_->resize(allocout);
			}
			tx=(*inarray_)[i].ix;
			ty=(*inarray_)[i].iy;
			tz=(*inarray_)[i].iz;
			for ( j=fast_one_shell_lowb[q]; j<fast_one_shell_highb[q]; j++ ) {
				tnv.ix=tx+nb[j][0];
				tnv.iy=ty+nb[j][1];
				tnv.iz=tz+nb[j][2];
				if ( tnv.ix<plength_ && tnv.ix>-1 &&
						tnv.iy<pwidth_ && tnv.iy>-1 &&
						tnv.iz<pheight_ && tnv.iz>-1 &&
						vp_[tnv.ix][tnv.iy][tnv.iz].inout &&
						!vp_[tnv.ix][tnv.iy][tnv.iz].isdone ) {
					boundpoint[tnv.ix][tnv.iy][tz+nb[j][2]].ix=boundpoint[tx][ty][tz].ix;
					boundpoint[tnv.ix][tnv.iy][tz+nb[j][2]].iy=boundpoint[tx][ty][tz].iy;
					boundpoint[tnv.ix][tnv.iy][tz+nb[j][2]].iz=boundpoint[tx][ty][tz].iz;
					dx=tnv.ix-boundpoint[tx][ty][tz].ix;
					dy=tnv.iy-boundpoint[tx][ty][tz].iy;
					dz=tnv.iz-boundpoint[tx][ty][tz].iz;
					squre=float(dx*dx+dy*dy+dz*dz);
					vp_[tnv.ix][tnv.iy][tnv.iz].distance=float(sqrt(squre));
					vp_[tnv.ix][tnv.iy][tnv.iz].isdone=true;
					vp_[tnv.ix][tnv.iy][tnv.iz].isbound=true;
					(*outarray_)[positout].ix=tnv.ix;
					(*outarray_)[positout].iy=tnv.iy;
					(*outarray_)[positout].iz=tnv.iz;
					positout++;eliminate++;
				} else if ( tnv.ix<plength_ && tnv.ix>-1 &&
						tnv.iy<pwidth_ && tnv.iy>-1 &&
						tnv.iz<pheight_ && tnv.iz>-1 &&
						vp_[tnv.ix][tnv.iy][tnv.iz].inout &&
						vp_[tnv.ix][tnv.iy][tnv.iz].isdone ) {

					dx=tnv.ix-boundpoint[tx][ty][tz].ix;
					dy=tnv.iy-boundpoint[tx][ty][tz].iy;
					dz=tnv.iz-boundpoint[tx][ty][tz].iz;
					squre=float(dx*dx+dy*dy+dz*dz);
					squre=float(sqrt(squre));
					if ( squre<vp_[tnv.ix][tnv.iy][tnv.iz].distance ) {
						boundpoint[tnv.ix][tnv.iy][tnv.iz].ix=boundpoint[tx][ty][tz].ix;
						boundpoint[tnv.ix][tnv.iy][tnv.iz].iy=boundpoint[tx][ty][tz].iy;
						boundpoint[tnv.ix][tnv.iy][tnv.iz].iz=boundpoint[tx][ty][tz].iz;
						vp_[tnv.ix][tnv.iy][tnv.iz].distance=float(squre);
						if ( !vp_[tnv.ix][tnv.iy][tnv.iz].isbound ) {
							vp_[tnv.ix][tnv.iy][tnv.iz].isbound=true;
							(*outarray_)[positout].ix=tnv.ix;
							(*outarray_)[positout].iy=tnv.iy;
							(*outarray_)[positout].iz=tnv.iz;
							positout++;
						}
					}

				}
			}
		}
	}

	outnum=positout;
	elimi=eliminate;

}

// Transform the vertices back to the input coordinates
void AtomicDepth::scale_surface() {
	for ( size_t i = 0; i < verts_.size(); i++ ) {
		verts_[i].x = verts_[i].x / scalefactor_ - ptran_.x;
		verts_[i].y = verts_[i].y / scalefactor_ - ptran_.y;
		verts_[i].z = verts_[i].z / scalefactor_ - ptran_.z;
	}
}



// Remove duplicate faces and vertices. Also join faces into larger continuous faces
void AtomicDepth::checkEuler()
{
	int vertnumber = verts_.size();
	int facenumber = faces_.size();
	int i,j,k;
	int ia,ib,ic;
	std::vector<std::vector<std::vector<int>>> vertdeg(4); //0 a 1 b 2 face 3 cutend]
	// int *vertdeg[4][20];//0 a 1 b 2 face 3 cutend]
	std::vector<bool> vertflag( int(vertnumber*1.1) );
	std::vector<int> vertgroup( int(vertnumber*1.1) );
	std::vector<int> vertnum( int(vertnumber*1.1) );
	// bool *vertflag=new bool[int(vertnumber*1.1)];
	// int *vertgroup=new int[int(vertnumber*1.1)];
	// int *vertnum=new int[int(vertnumber*1.1)];
	for(i=0;i<4;i++)
	{
		vertdeg[i].resize(20);
		for(j=0;j<20;j++) {
			vertdeg[i][j].resize(int(vertnumber*1.1));
			// vertdeg[i][j]=new int[int(vertnumber*1.1)];
		}
	}
	for(j=0;j<vertnumber;j++)
	{
		vertnum[j]=0;
		vertgroup[j]=0;
		vertflag[j]=true;
	}
	std::vector<bool> flagface(facenumber);
	// bool *flagface=new bool[facenumber];
	//degree of each vert
	for(i=0;i<facenumber;i++)
	{
		ia=faces_[i].a;
		ib=faces_[i].b;
		ic=faces_[i].c;
		vertdeg[0][vertnum[ia]][ia]=ib;
		vertdeg[1][vertnum[ia]][ia]=ic;
		vertdeg[2][vertnum[ia]][ia]=i;
		vertdeg[0][vertnum[ib]][ib]=ic;
		vertdeg[1][vertnum[ib]][ib]=ia;
		vertdeg[2][vertnum[ib]][ib]=i;
		vertdeg[0][vertnum[ic]][ic]=ia;
		vertdeg[1][vertnum[ic]][ic]=ib;
		vertdeg[2][vertnum[ic]][ic]=i;
		vertnum[ia]++;
		vertnum[ib]++;
		vertnum[ic]++;
		flagface[i]=true;	
	}//i
	int jb,jc;
	int kb,kc;
	int l,m;
	// vertinfo *dupvert;
	int numdup=0;
	int allocdup=20;
	std::vector<vertinfo> dupvert(allocdup);
	// dupvert=new vertinfo[allocdup];
	bool flagdup;
	std::vector<std::vector<int>> tpindex(3);
	// int *tpindex[3];
	for(i=0;i<3;i++)
	{
		tpindex[i].resize(20);
		// tpindex[i]=new int[20];
	}
	for(i=0;i<vertnumber;i++)
	{
		//remove dup faces
		for(j=0;j<vertnum[i]-1;j++)
		{
			jb=vertdeg[0][j][i];
			jc=vertdeg[1][j][i];
			for(k=j+1;k<vertnum[i];k++)
			{
				kb=vertdeg[0][k][i];
				kc=vertdeg[1][k][i];
				if(jb==kc && jc==kb)
				{
		//			printf("%d dup face %d %d [%d %d %d]\n",i,vertdeg[2][j][i],vertdeg[2][k][i],i,jb,jc);
					flagface[vertdeg[2][j][i]]=false;
					flagface[vertdeg[2][k][i]]=false;
					for(l=j;l<k-1;l++)
					{
						vertdeg[0][l][i]=vertdeg[0][l+1][i];
						vertdeg[1][l][i]=vertdeg[1][l+1][i];
						vertdeg[2][l][i]=vertdeg[2][l+1][i];
					}
					for(l=k-1;l<vertnum[i]-2;l++)
					{
						vertdeg[0][l][i]=vertdeg[0][l+2][i];
						vertdeg[1][l][i]=vertdeg[1][l+2][i];
						vertdeg[2][l][i]=vertdeg[2][l+2][i];
					}
					j--;
					k=vertnum[i];
					vertnum[i]-=2;
				}//duplicate
				else if(jb==kb && jc==kc)
				{
		//			printf("wrong same faces %d %d\n",vertdeg[2][j][i],vertdeg[2][k][i]);
				}
			}//k
		}//j
		if(vertnum[i]==0)
		{
	//		printf("no use vertex %d\n",i);
			vertflag[i]=false;
			continue;
		}
		else if(vertnum[i]==1 || vertnum[i]==2)
		{
		//	printf("single vertex %d %d \n",i,vertnum[i]);
		//	vertflag[i]=false;
		}
		//reorder 
		flagdup=false;
		for(j=0;j<vertnum[i]-1;j++)
		{
			for(k=j+1;k<vertnum[i];k++)
			{
				if(vertdeg[0][j][i]==vertdeg[0][k][i])
				{
					flagdup=true;
					break;
				}
			}
			if(flagdup)
				break;
		}
		if(flagdup)
		{
			for(k=j;k<vertnum[i];k++)
			{
				tpindex[0][k-j]=vertdeg[0][k][i];
				tpindex[1][k-j]=vertdeg[1][k][i];
				tpindex[2][k-j]=vertdeg[2][k][i];
			}
			for(k=0;k<j;k++)
			{
				tpindex[0][vertnum[i]-j+k]=vertdeg[0][k][i];
				tpindex[1][vertnum[i]-j+k]=vertdeg[1][k][i];
				tpindex[2][vertnum[i]-j+k]=vertdeg[2][k][i];
			}
			for(k=0;k<vertnum[i];k++)
			{
				vertdeg[0][k][i]=tpindex[0][k];
				vertdeg[1][k][i]=tpindex[1][k];
				vertdeg[2][k][i]=tpindex[2][k];
			}
		}
		//arrage all faces around a vert
		j=0;	
		while(j<vertnum[i])//start cycle
		{		
			jb=vertdeg[0][j][i];
			jc=vertdeg[1][j][i];
			m=j;
			do{//find m+1
				k=vertnum[i];
				for(k=m+1;k<vertnum[i];k++)
				{
					if(vertdeg[0][k][i]==jc)
						break;
				}
				if(k<vertnum[i])
				{
					if(k!=m+1)
					{
						l=vertdeg[0][m+1][i];
						vertdeg[0][m+1][i]=vertdeg[0][k][i];
						vertdeg[0][k][i]=l;
						l=vertdeg[1][m+1][i];
						vertdeg[1][m+1][i]=vertdeg[1][k][i];
						vertdeg[1][k][i]=l;
						l=vertdeg[2][m+1][i];
						vertdeg[2][m+1][i]=vertdeg[2][k][i];
						vertdeg[2][k][i]=l;
					}
					jc=vertdeg[1][m+1][i];	
					m++;
				}
				else
				{
					break;
				}
			}while(jc!=jb && m<vertnum[i]);
			if(jc==jb)//one cycle
			{
				vertdeg[3][vertgroup[i]][i]=m;
				vertgroup[i]++;
			}
			else//single
			{
		//		printf("no corre index %d %d\n",i,jc);
				vertdeg[3][vertgroup[i]][i]=m;
				vertgroup[i]++;		
				for(j=0;j<vertnum[i];j++)
				{
		//			printf("detail %d %d %d %d %d\n",j,vertdeg[0][j][i],vertdeg[1][j][i],vertdeg[2][j][i],vertdeg[3][j][i]);
				}
			}
			j=m+1;
		}//while
		if(vertgroup[i]!=1)
		{
	//		printf("split vert %d %d\n",i,vertgroup[i]);
	//		for(j=0;j<vertnum[i];j++)
	//		{
	//			printf("%d %d %d %d %d\n",j,vertdeg[0][j][i],vertdeg[1][j][i],vertdeg[2][j][i],vertdeg[3][j][i]);
	//		}
			if(numdup+vertgroup[i]>allocdup)
			{
				allocdup*=2;
				dupvert.resize(allocdup);
				// dupvert=(vertinfo *)doubleloc(dupvert,allocdup*sizeof(vertinfo));
			}
			
			for(j=1;j<vertgroup[i];j++)
			{	
				dupvert[numdup]=verts_[i];
				vertflag[numdup+vertnumber]=true;
				vertgroup[numdup+vertnumber]=1;
				vertnum[numdup+vertnumber]=vertdeg[3][j][i]-vertdeg[3][j-1][i];
				for(k=0;k<vertnum[numdup+vertnumber];k++)
				{
					vertdeg[0][k][numdup+vertnumber]=vertdeg[0][vertdeg[3][j-1][i]+k+1][numdup+vertnumber];
					vertdeg[1][k][numdup+vertnumber]=vertdeg[1][vertdeg[3][j-1][i]+k+1][numdup+vertnumber];
					vertdeg[2][k][numdup+vertnumber]=vertdeg[2][vertdeg[3][j-1][i]+k+1][numdup+vertnumber];
				}
				for(k=vertdeg[3][j-1][i]+1;k<=vertdeg[3][j][i];k++)
				{
		//			printf("changing %d %d\n",k,vertdeg[2][k][i]);
					if(faces_[vertdeg[2][k][i]].a==i)
					{
						faces_[vertdeg[2][k][i]].a=numdup+vertnumber;
						m=faces_[vertdeg[2][k][i]].b;
						for(l=0;l<vertnum[m];l++)
						{
							if(vertdeg[2][l][m]==vertdeg[2][k][i])
							{
								if(vertdeg[0][l][m]==i)
								{
									vertdeg[0][l][m]=numdup+vertnumber;
								}
								else if(vertdeg[1][l][m]==i)
								{
									vertdeg[1][l][m]=numdup+vertnumber;
								}
								else 
								{
							//		printf("wrong modified vertab %d\n",m);
								}
							}
						}
						m=faces_[vertdeg[2][k][i]].c;
						for(l=0;l<vertnum[m];l++)
						{
							if(vertdeg[2][l][m]==vertdeg[2][k][i])
							{
								if(vertdeg[0][l][m]==i)
								{
									vertdeg[0][l][m]=numdup+vertnumber;
								}
								else if(vertdeg[1][l][m]==i)
								{
									vertdeg[1][l][m]=numdup+vertnumber;
								}
								else 
								{
									std::cout << "wrong modified vertac " << m << std::endl;;
								}
							}
						}
					}
					else if(faces_[vertdeg[2][k][i]].b==i)
					{
						faces_[vertdeg[2][k][i]].b=numdup+vertnumber;
						m=faces_[vertdeg[2][k][i]].a;
						for(l=0;l<vertnum[m];l++)
						{
							if(vertdeg[2][l][m]==vertdeg[2][k][i])
							{
								if(vertdeg[0][l][m]==i)
								{
									vertdeg[0][l][m]=numdup+vertnumber;
								}
								else if(vertdeg[1][l][m]==i)
								{
									vertdeg[1][l][m]=numdup+vertnumber;
								}
								else 
								{
							//		printf("wrong modified vertba %d\n",m);
								}
							}
						}
						m=faces_[vertdeg[2][k][i]].c;
						for(l=0;l<vertnum[m];l++)
						{
							if(vertdeg[2][l][m]==vertdeg[2][k][i])
							{
								if(vertdeg[0][l][m]==i)
								{
									vertdeg[0][l][m]=numdup+vertnumber;
								}
								else if(vertdeg[1][l][m]==i)
								{
									vertdeg[1][l][m]=numdup+vertnumber;
								}
								else 
								{
							//		printf("wrong modified vertbc %d\n",m);
								}
							}
						}
					}
					else if(faces_[vertdeg[2][k][i]].c==i)
					{
						faces_[vertdeg[2][k][i]].c=numdup+vertnumber;
						m=faces_[vertdeg[2][k][i]].a;
						for(l=0;l<vertnum[m];l++)
						{
							if(vertdeg[2][l][m]==vertdeg[2][k][i])
							{
								if(vertdeg[0][l][m]==i)
								{
									vertdeg[0][l][m]=numdup+vertnumber;
								}
								else if(vertdeg[1][l][m]==i)
								{
									vertdeg[1][l][m]=numdup+vertnumber;
								}
								else 
								{
							//		printf("wrong modified vertca %d\n",m);
								}
							}
						}
						m=faces_[vertdeg[2][k][i]].b;
						for(l=0;l<vertnum[m];l++)
						{
							if(vertdeg[2][l][m]==vertdeg[2][k][i])
							{
								if(vertdeg[0][l][m]==i)
								{
									vertdeg[0][l][m]=numdup+vertnumber;
								}
								else if(vertdeg[1][l][m]==i)
								{
									vertdeg[1][l][m]=numdup+vertnumber;
								}
								else 
								{
						//			printf("wrong modified vertcb %d\n",m);
								}
							}
						}
					}
					else
					{
					//	printf("wrong vert %d face %d [%d %d %d]\n",i,vertdeg[2][k][i],faces[vertdeg[2][k][i]].a,
					//		faces[vertdeg[2][k][i]].b,faces[vertdeg[2][k][i]].c);
					}
				}//k
				vertgroup[i]=1;
				numdup++;
			}//j  
		}//if need to split
	}//i
	// for(i=0;i<3;i++)
	// {
	// 	delete[]tpindex[i];
	// }
	//reduce face
	int totfacenew=0;
	for(i=0;i<facenumber;i++)
	{
		if(flagface[i] && totfacenew!=i)
		{
			faces_[totfacenew]=faces_[i];
			totfacenew++;
		}
		else if(flagface[i])
		{
			totfacenew++;
		}
	}
//	printf("number faces from %d to %d\n",facenumber,totfacenew);
	facenumber=totfacenew;
	
	//new points
	int totvertnew=0;
	std::vector<int> vertindex2(vertnumber+numdup);
	// int *vertindex2=new int[vertnumber+numdup];
	verts_.resize(vertnumber+numdup);
	// verts=(vertinfo*)doubleloc(verts,(vertnumber+numdup)*sizeof(vertinfo));
	for(i=0;i<vertnumber;i++)
	{
		if(vertflag[i] && totvertnew!=i)
		{
			vertindex2[i]=totvertnew;
			verts_[totvertnew]=verts_[i];
			totvertnew++;
		}
		else if(vertflag[i])
		{
			vertindex2[i]=totvertnew;
			totvertnew++;	
		}
	}
	for(i=0;i<numdup;i++)
	{
		vertindex2[vertnumber+i]=totvertnew;
		verts_[totvertnew]=dupvert[i];
		totvertnew++;
	}
	for(i=0;i<facenumber;i++)
	{
		faces_[i].a=vertindex2[faces_[i].a];
		faces_[i].b=vertindex2[faces_[i].b];
		faces_[i].c=vertindex2[faces_[i].c];
	}
	// delete[]vertindex2;
//	printf("number verts from %d to %d (new added %d)\n",vertnumber,totvertnew,numdup);
	vertnumber=totvertnew;

	faces_.resize(facenumber);
	verts_.resize(vertnumber);
	// if((2*vertnumber-facenumber)%4!=0)
	// printf("euler num %d\n",2*vertnumber-facenumber);//comp+cav-genus

	//release
	// for(i=0;i<4;i++)
	// {
	// 	for(j=0;j<20;j++)
	// 		delete[]vertdeg[i][j];
	// }
	// delete[]vertgroup;
	// delete[]vertflag;
	// delete[]vertnum;
	// delete[]flagface;
	// delete[]dupvert;
}

// Calculate normals of vertices and faces
// Normals of vertices are just averages of faces
void AtomicDepth::computenorm()
{
	size_t i;
	double pnorm;
	for(i=0;i<verts_.size();i++)
	{
		verts_[i].pn.x=0;
		verts_[i].pn.y=0;
		verts_[i].pn.z=0;
	}
	point3d p1,p2,p3;
	point3d p12,p13;
	point3d pn;
	for(i=0;i<faces_.size();i++)
	{
		p1.x=verts_[faces_[i].a].x;
		p1.y=verts_[faces_[i].a].y;
		p1.z=verts_[faces_[i].a].z;
		p2.x=verts_[faces_[i].b].x;
		p2.y=verts_[faces_[i].b].y;
		p2.z=verts_[faces_[i].b].z;
		p3.x=verts_[faces_[i].c].x;
		p3.y=verts_[faces_[i].c].y;
		p3.z=verts_[faces_[i].c].z;
		p12.x=p2.x-p1.x;
		p12.y=p2.y-p1.y;
		p12.z=p2.z-p1.z;
		p13.x=p3.x-p1.x;
		p13.y=p3.y-p1.y;
		p13.z=p3.z-p1.z;
		pn.x=p12.y*p13.z-p12.z*p13.y;
		pn.y=p12.z*p13.x-p12.x*p13.z;
		pn.z=p12.x*p13.y-p12.y*p13.x;
		faces_[i].area=0.5*sqrt(pn.x*pn.x+pn.y*pn.y+pn.z*pn.z);
		faces_[i].pn.x=0.5*pn.x/faces_[i].area;
		faces_[i].pn.y=0.5*pn.y/faces_[i].area;
		faces_[i].pn.z=0.5*pn.z/faces_[i].area;
		//*
		//without area
		verts_[faces_[i].a].pn.x+=faces_[i].pn.x;
		verts_[faces_[i].a].pn.y+=faces_[i].pn.y;
		verts_[faces_[i].a].pn.z+=faces_[i].pn.z;
		verts_[faces_[i].b].pn.x+=faces_[i].pn.x;
		verts_[faces_[i].b].pn.y+=faces_[i].pn.y;
		verts_[faces_[i].b].pn.z+=faces_[i].pn.z;
		verts_[faces_[i].c].pn.x+=faces_[i].pn.x;
		verts_[faces_[i].c].pn.y+=faces_[i].pn.y;
		verts_[faces_[i].c].pn.z+=faces_[i].pn.z;
		//*/
		/*
		//with area
		verts_[faces_[i].a].pn.x+=pn.x;
		verts_[faces_[i].a].pn.y+=pn.y;
		verts_[faces_[i].a].pn.z+=pn.z;
		verts_[faces_[i].b].pn.x+=pn.x;
		verts_[faces_[i].b].pn.y+=pn.y;
		verts_[faces_[i].b].pn.z+=pn.z;
		verts_[faces_[i].c].pn.x+=pn.x;
		verts_[faces_[i].c].pn.y+=pn.y;
		verts_[faces_[i].c].pn.z+=pn.z;
		*/
	}
	for(i=0;i<verts_.size();i++)
	{
		pn.x=verts_[i].pn.x;
		pn.y=verts_[i].pn.y;
		pn.z=verts_[i].pn.z;
		pnorm=std::sqrt(pn.x*pn.x+pn.y*pn.y+pn.z*pn.z);
		if(pnorm==0.0)
		{
			pn.x=0.0;
			pn.y=0.0;
			pn.z=0.0;
			verts_[i].pn.x=pn.x;
			verts_[i].pn.y=pn.y;
			verts_[i].pn.z=pn.z;
			continue;
		}
		pn.x/=pnorm;
		pn.y/=pnorm;
		pn.z/=pnorm;
		verts_[i].pn.x=pn.x;
		verts_[i].pn.y=pn.y;
		verts_[i].pn.z=pn.z;
	}
}


// Prepare for marching cube protocol
void AtomicDepth::marchingcubeinit(int stype)
{
	int i,j,k;
	//vdw
	if(stype==1)
	{
		for(i=0;i<plength_;i++)
		{
			for(j=0;j<pwidth_;j++)
			{
				for(k=0;k<pheight_;k++)
				{
					vp_[i][j][k].isbound=false;
				}
			}
		}

	}
	//ses
	else if(stype==4)
	{
		///////////////without vdw
		for(i=0;i<plength_;i++)
		{
			for(j=0;j<pwidth_;j++)
			{
				for(k=0;k<pheight_;k++)
				{
					vp_[i][j][k].isdone=false;
					if(vp_[i][j][k].isbound)
					{
						vp_[i][j][k].isdone=true;	
					}
					//new add
					vp_[i][j][k].isbound=false;
				}
			}
		}
		
	}
	else if(stype==2)
	{	
		///////////////////////after vdw
		for(i=0;i<plength_;i++)
		{
			for(j=0;j<pwidth_;j++)
			{
				for(k=0;k<pheight_;k++)
				{
				//	if(vp_[i][j][k].inout && vp_[i][j][k].distance>=cutradis)
					if(vp_[i][j][k].isbound && vp_[i][j][k].isdone)
					{
						vp_[i][j][k].isbound=false;
					}
					else if(vp_[i][j][k].isbound && !vp_[i][j][k].isdone)
					{
						vp_[i][j][k].isdone=true;	
					}
				}
			}
		}
		
	}
	//sas
	else if(stype==3)
	{
		for(i=0;i<plength_;i++)
		{
			for(j=0;j<pwidth_;j++)
			{
				for(k=0;k<pheight_;k++)
				{
					vp_[i][j][k].isbound=false;
				}
			}
		}
	}
	
}

// Smooth the surface
void AtomicDepth::laplaciansmooth(int numiter)
{
	std::vector<point3d> tps(verts_.size());
	// point3d *tps=new point3d[vertnumber];
	std::vector<std::vector<int>> vertdeg(20);
	// int *vertdeg[20];
	size_t i;
	int j;
	bool flagvert;
	for(i=0;i<20;i++)
	{
		vertdeg[i].resize(verts_.size());
		// vertdeg[i]=new int[vertnumber];		
	}
	for(i=0;i<verts_.size();i++)
	{
		vertdeg[0][i]=0;
	}
	for(i=0;i<faces_.size();i++)
	{
		//a
		flagvert=true;
		for(j=0;j<vertdeg[0][faces_[i].a];j++)
		{
			if(faces_[i].b==vertdeg[j+1][faces_[i].a])
			{
				flagvert=false;
				break;
			}
		}
		if(flagvert)
		{
			vertdeg[0][faces_[i].a]++;
			vertdeg[vertdeg[0][faces_[i].a]][faces_[i].a]=faces_[i].b;
			
		}
		flagvert=true;
		for(j=0;j<vertdeg[0][faces_[i].a];j++)
		{
			if(faces_[i].c==vertdeg[j+1][faces_[i].a])
			{
				flagvert=false;
				break;
			}
		}
		if(flagvert)
		{
			vertdeg[0][faces_[i].a]++;
			vertdeg[vertdeg[0][faces_[i].a]][faces_[i].a]=faces_[i].c;
			
		}
		//b
		flagvert=true;
		for(j=0;j<vertdeg[0][faces_[i].b];j++)
		{
			if(faces_[i].a==vertdeg[j+1][faces_[i].b])
			{
				flagvert=false;
				break;
			}
		}
		if(flagvert)
		{
			vertdeg[0][faces_[i].b]++;
			vertdeg[vertdeg[0][faces_[i].b]][faces_[i].b]=faces_[i].a;
			
		}
		flagvert=true;
		for(j=0;j<vertdeg[0][faces_[i].b];j++)
		{
			if(faces_[i].c==vertdeg[j+1][faces_[i].b])
			{
				flagvert=false;
				break;
			}
		}
		if(flagvert)
		{
			vertdeg[0][faces_[i].b]++;
			vertdeg[vertdeg[0][faces_[i].b]][faces_[i].b]=faces_[i].c;
			
		}
		//c
		flagvert=true;
		for(j=0;j<vertdeg[0][faces_[i].c];j++)
		{
			if(faces_[i].a==vertdeg[j+1][faces_[i].c])
			{
				flagvert=false;
				break;
			}
		}
		if(flagvert)
		{
			vertdeg[0][faces_[i].c]++;
			vertdeg[vertdeg[0][faces_[i].c]][faces_[i].c]=faces_[i].a;
			
		}
		flagvert=true;
		for(j=0;j<vertdeg[0][faces_[i].c];j++)
		{
			if(faces_[i].b==vertdeg[j+1][faces_[i].c])
			{
				flagvert=false;
				break;
			}
		}
		if(flagvert)
		{
			vertdeg[0][faces_[i].c]++;
			vertdeg[vertdeg[0][faces_[i].c]][faces_[i].c]=faces_[i].b;
			
		}
	}
	
	double wt=1.00;
	double wt2=0.50;
	int ssign;
	int k;
	double outwt=0.75/(scalefactor_+3.5);//area-preserving
for(k=0;k<numiter;k++)
{
	for(i=0;i<verts_.size();i++)
	{
		if(vertdeg[0][i]<3)
		{
			tps[i].x=verts_[i].x;
			tps[i].y=verts_[i].y;
			tps[i].z=verts_[i].z;
		}
		else if(vertdeg[0][i]==3 || vertdeg[0][i]==4)
		{
			tps[i].x=0;
			tps[i].y=0;
			tps[i].z=0;
			for(j=0;j<vertdeg[0][i];j++)
			{
				tps[i].x+=verts_[vertdeg[j+1][i]].x;
				tps[i].y+=verts_[vertdeg[j+1][i]].y;
				tps[i].z+=verts_[vertdeg[j+1][i]].z;
			}
			tps[i].x+=wt2*verts_[i].x;
			tps[i].y+=wt2*verts_[i].y;
			tps[i].z+=wt2*verts_[i].z;
			tps[i].x/=float(wt2+vertdeg[0][i]);
			tps[i].y/=float(wt2+vertdeg[0][i]);
			tps[i].z/=float(wt2+vertdeg[0][i]);
		}
		else
		{
			tps[i].x=0;
			tps[i].y=0;
			tps[i].z=0;
			for(j=0;j<vertdeg[0][i];j++)
			{
				tps[i].x+=verts_[vertdeg[j+1][i]].x;
				tps[i].y+=verts_[vertdeg[j+1][i]].y;
				tps[i].z+=verts_[vertdeg[j+1][i]].z;
			}
			tps[i].x+=wt*verts_[i].x;
			tps[i].y+=wt*verts_[i].y;
			tps[i].z+=wt*verts_[i].z;
			tps[i].x/=float(wt+vertdeg[0][i]);
			tps[i].y/=float(wt+vertdeg[0][i]);
			tps[i].z/=float(wt+vertdeg[0][i]);
		}
	}
	for(i=0;i<verts_.size();i++)
	{
		verts_[i].x=tps[i].x;
		verts_[i].y=tps[i].y;
		verts_[i].z=tps[i].z;
	}
	computenorm();
	for(i=0;i<verts_.size();i++)
	{
		if(verts_[i].inout) ssign=1;
		else ssign=-1;
		verts_[i].x+=ssign*outwt*verts_[i].pn.x;
		verts_[i].y+=ssign*outwt*verts_[i].pn.y;
		verts_[i].z+=ssign*outwt*verts_[i].pn.z;
	}
}

}


// perform marching cube protocol. Known as Vertex-Connected Marching Cubes (VCMC)
void AtomicDepth::marchingcube(int stype)
{
	int i,j,k;
	marchingcubeinit(stype);
	std::vector<std::vector<std::vector<int>>> vertseq(plength_);
	// int ***vertseq;
	// vertseq=new int**[plength_];
	for(i=0;i<plength_;i++)
	{
		vertseq[i].resize(pwidth_);
		// vertseq[i]=new int*[pwidth_];
	}
	for(i=0;i<plength_;i++)
	{
		for(j=0;j<pwidth_;j++)
		{
			vertseq[i][j].resize(pheight_);
			// vertseq[i][j]=new int[pheight_];
		}
	}
	for(i=0;i<plength_;i++)
	{
		for(j=0;j<pwidth_;j++)
		{
			for(k=0;k<pheight_;k++)
			{
				vertseq[i][j][k]=-1;
			}		
		}
	}
	faces_.clear();
	verts_.clear();
//	int allocface=20;
//	int allocvert=12;
	int allocvert=4*(pheight_*plength_+pwidth_*plength_+pheight_*pwidth_);
	int allocface=2*allocvert;
	int facenumber=0;
	int vertnumber=0;
	verts_.resize(allocvert);
	faces_.resize(allocface);
	// verts=new vertinfo[allocvert];
	// faces=new faceinfo[allocface];
	
	
	int sumtype;
	int ii,jj,kk;
	// int tp[6][3];
	std::vector<std::vector<int>> tp(6, {0, 0, 0});
	/////////////////////////////////////////new added  normal is outer
	//face1
	for(i=0;i<1;i++)
	{
		for(j=0;j<pwidth_-1;j++)
		{
			for(k=0;k<pheight_-1;k++)
			{
				if(vp_[i][j][k].isdone && vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone
					&& vp_[i][j][k+1].isdone)
				{
					tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
					tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
					tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;
				    tp[3][0]=i;tp[3][1]=j;tp[3][2]=k+1;
					for(ii=0;ii<4;ii++)
					{
						if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
						{
							vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
							verts_[vertnumber].x=tp[ii][0];
							verts_[vertnumber].y=tp[ii][1];
							verts_[vertnumber].z=tp[ii][2];
							vertnumber++;
						}
					}
					faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
					faces_[facenumber].c=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
					faces_[facenumber++].b=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
					faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
					faces_[facenumber].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
					faces_[facenumber++].b=vertseq[tp[3][0]][tp[3][1]][tp[3][2]];
				}
				else if((vp_[i][j][k].isdone && vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone)
					||( vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone && vp_[i][j][k+1].isdone)
					||( vp_[i][j+1][k+1].isdone && vp_[i][j][k+1].isdone && vp_[i][j][k].isdone)
					||(vp_[i][j][k+1].isdone && vp_[i][j][k].isdone && vp_[i][j+1][k].isdone))
				{
					if(vp_[i][j][k].isdone && vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone)
					{
						tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
						tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
						tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;
					}
				    else if( vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone && vp_[i][j][k+1].isdone)
					{
						tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
						tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;
						tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;
					}
					else if( vp_[i][j+1][k+1].isdone && vp_[i][j][k+1].isdone && vp_[i][j][k].isdone)
					{
						tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
						tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
						tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
					}
					else if(vp_[i][j][k+1].isdone && vp_[i][j][k].isdone && vp_[i][j+1][k].isdone)
					{
						tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
						tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
						tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
					}
					for(ii=0;ii<3;ii++)
					{
						if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
						{
							vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
							verts_[vertnumber].x=tp[ii][0];
							verts_[vertnumber].y=tp[ii][1];
							verts_[vertnumber].z=tp[ii][2];
							vertnumber++;
						}
					}
					faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
					faces_[facenumber].c=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
					faces_[facenumber++].b=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
				}
				
			}
		}
	}
	//face3
	for(i=0;i<plength_-1;i++)
	{
		for(j=0;j<1;j++)
		{
			for(k=0;k<pheight_-1;k++)
			{
				if(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone
					&& vp_[i][j][k+1].isdone)
				{
					tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
					tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
					tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
					tp[3][0]=i;tp[3][1]=j;tp[3][2]=k+1;
					for(ii=0;ii<4;ii++)
					{
						if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
						{
							vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
							verts_[vertnumber].x=tp[ii][0];
							verts_[vertnumber].y=tp[ii][1];
							verts_[vertnumber].z=tp[ii][2];
							vertnumber++;
						}
					}
					faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
					faces_[facenumber].b=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
					faces_[facenumber++].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
					faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
					faces_[facenumber].b=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
					faces_[facenumber++].c=vertseq[tp[3][0]][tp[3][1]][tp[3][2]];
				}
				else if((vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone)
					||( vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone && vp_[i][j][k+1].isdone)
					||( vp_[i+1][j][k+1].isdone && vp_[i][j][k+1].isdone && vp_[i][j][k].isdone)
					||(vp_[i][j][k+1].isdone && vp_[i][j][k].isdone && vp_[i+1][j][k].isdone))
				{
					if(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone)
					{
						tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
						tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
						tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
					}
					else if( vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone && vp_[i][j][k+1].isdone)
					{
						tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
						tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
						tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;
					}
					else if( vp_[i+1][j][k+1].isdone && vp_[i][j][k+1].isdone && vp_[i][j][k].isdone)
					{
						tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
						tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
						tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
					}
					else if(vp_[i][j][k+1].isdone && vp_[i][j][k].isdone && vp_[i+1][j][k].isdone)
					{
						tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
						tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
						tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
					}
					for(ii=0;ii<3;ii++)
					{
						if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
						{
							vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
							verts_[vertnumber].x=tp[ii][0];
							verts_[vertnumber].y=tp[ii][1];
							verts_[vertnumber].z=tp[ii][2];
							vertnumber++;
						}
					}
					faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
					faces_[facenumber].b=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
					faces_[facenumber++].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
				}
				
			}
		}
	}
	//face5
	for(i=0;i<plength_-1;i++)
	{
		for(j=0;j<pwidth_-1;j++)
		{
			for(k=0;k<1;k++)
			{
				if(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j+1][k].isdone
					&& vp_[i][j+1][k].isdone)
				{
					tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
					tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
					tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
					tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k;
					for(ii=0;ii<4;ii++)
					{
						if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
						{
							vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
							verts_[vertnumber].x=tp[ii][0];
							verts_[vertnumber].y=tp[ii][1];
							verts_[vertnumber].z=tp[ii][2];
							vertnumber++;
						}
					}
					faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
					faces_[facenumber].c=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
					faces_[facenumber++].b=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
					faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
					faces_[facenumber].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
					faces_[facenumber++].b=vertseq[tp[3][0]][tp[3][1]][tp[3][2]];
				}
				else if((vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j+1][k].isdone)
					||( vp_[i+1][j][k].isdone && vp_[i+1][j+1][k].isdone && vp_[i][j+1][k].isdone)
					||( vp_[i+1][j+1][k].isdone && vp_[i][j+1][k].isdone && vp_[i][j][k].isdone)
					||(vp_[i][j+1][k].isdone && vp_[i][j][k].isdone && vp_[i+1][j][k].isdone))
				{
					if(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j+1][k].isdone)
					{
						tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
						tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
						tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
					}
					else if( vp_[i+1][j][k].isdone && vp_[i+1][j+1][k].isdone && vp_[i][j+1][k].isdone)
					{
						tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
						tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
						tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
					}
					else if( vp_[i+1][j+1][k].isdone && vp_[i][j+1][k].isdone && vp_[i][j][k].isdone)
					{
						tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k;
						tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
						tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
					}
					else if(vp_[i][j+1][k].isdone && vp_[i][j][k].isdone && vp_[i+1][j][k].isdone)
					{
						tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
						tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
						tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
					}
					for(ii=0;ii<3;ii++)
					{
						if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
						{
							vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
							verts_[vertnumber].x=tp[ii][0];
							verts_[vertnumber].y=tp[ii][1];
							verts_[vertnumber].z=tp[ii][2];
							vertnumber++;
						}
					}
					faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
					faces_[facenumber].c=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
					faces_[facenumber++].b=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
				}
				
			}
		}
	}
	//face2
	for(i=plength_-1;i<plength_;i++)
	{
		for(j=0;j<pwidth_-1;j++)
		{
			for(k=0;k<pheight_-1;k++)
			{
				if(vp_[i][j][k].isdone && vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone
					&& vp_[i][j][k+1].isdone)
				{
					tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
					tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
					tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;
					tp[3][0]=i;tp[3][1]=j;tp[3][2]=k+1;
					for(ii=0;ii<4;ii++)
					{
						if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
						{
							vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
							verts_[vertnumber].x=tp[ii][0];
							verts_[vertnumber].y=tp[ii][1];
							verts_[vertnumber].z=tp[ii][2];
							vertnumber++;
						}
					}
					faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
					faces_[facenumber].b=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
					faces_[facenumber++].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
					faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
					faces_[facenumber].b=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
					faces_[facenumber++].c=vertseq[tp[3][0]][tp[3][1]][tp[3][2]];
				}
				else if((vp_[i][j][k].isdone && vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone)
					||( vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone && vp_[i][j][k+1].isdone)
					||( vp_[i][j+1][k+1].isdone && vp_[i][j][k+1].isdone && vp_[i][j][k].isdone)
					||(vp_[i][j][k+1].isdone && vp_[i][j][k].isdone && vp_[i][j+1][k].isdone))
				{
					if(vp_[i][j][k].isdone && vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone)
					{
						tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
						tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
						tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;
					}
					else if( vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone && vp_[i][j][k+1].isdone)
					{
						tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
						tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;
						tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;
					}
					else if( vp_[i][j+1][k+1].isdone && vp_[i][j][k+1].isdone && vp_[i][j][k].isdone)
					{
						tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
						tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
						tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
					}
					else if(vp_[i][j][k+1].isdone && vp_[i][j][k].isdone && vp_[i][j+1][k].isdone)
					{
						tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
						tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
						tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
					}
					for(ii=0;ii<3;ii++)
					{
						if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
						{
							vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
							verts_[vertnumber].x=tp[ii][0];
							verts_[vertnumber].y=tp[ii][1];
							verts_[vertnumber].z=tp[ii][2];
							vertnumber++;
						}
					}
					faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
					faces_[facenumber].b=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
					faces_[facenumber++].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
				}
				
			}
		}
	}
	//face4
	for(i=0;i<plength_-1;i++)
	{
		for(j=pwidth_-1;j<pwidth_;j++)
		{
			for(k=0;k<pheight_-1;k++)
			{
				if(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone
					&& vp_[i][j][k+1].isdone)
				{
					tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
					tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
					tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
					tp[3][0]=i;tp[3][1]=j;tp[3][2]=k+1;
					for(ii=0;ii<4;ii++)
					{
						if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
						{
							vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
							verts_[vertnumber].x=tp[ii][0];
							verts_[vertnumber].y=tp[ii][1];
							verts_[vertnumber].z=tp[ii][2];
							vertnumber++;
						}
					}
					faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
					faces_[facenumber].c=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
					faces_[facenumber++].b=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
					faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
					faces_[facenumber].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
					faces_[facenumber++].b=vertseq[tp[3][0]][tp[3][1]][tp[3][2]];
				}
				else if((vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone)
					||( vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone && vp_[i][j][k+1].isdone)
					||( vp_[i+1][j][k+1].isdone && vp_[i][j][k+1].isdone && vp_[i][j][k].isdone)
					||(vp_[i][j][k+1].isdone && vp_[i][j][k].isdone && vp_[i+1][j][k].isdone))
				{
					if(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone)
					{
						tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
						tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
						tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
					}
					else if( vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone && vp_[i][j][k+1].isdone)
					{
						tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
						tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
						tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;
					}
					else if( vp_[i+1][j][k+1].isdone && vp_[i][j][k+1].isdone && vp_[i][j][k].isdone)
					{
						tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
						tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
						tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
					}
					else if(vp_[i][j][k+1].isdone && vp_[i][j][k].isdone && vp_[i+1][j][k].isdone)
					{
						tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
						tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
						tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
					}
					for(ii=0;ii<3;ii++)
					{
						if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
						{
							vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
							verts_[vertnumber].x=tp[ii][0];
							verts_[vertnumber].y=tp[ii][1];
							verts_[vertnumber].z=tp[ii][2];
							vertnumber++;
						}
					}
					faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
					faces_[facenumber].c=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
					faces_[facenumber++].b=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
				}
				
			}
		}
	}
	//face6
	for(i=0;i<plength_-1;i++)
	{
		for(j=0;j<pwidth_-1;j++)
		{
			for(k=pheight_-1;k<pheight_;k++)
			{
				if(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j+1][k].isdone
					&& vp_[i][j+1][k].isdone)
				{
					tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
					tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
					tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
					tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k;
					for(ii=0;ii<4;ii++)
					{
						if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
						{
							vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
							verts_[vertnumber].x=tp[ii][0];
							verts_[vertnumber].y=tp[ii][1];
							verts_[vertnumber].z=tp[ii][2];
							vertnumber++;
						}
					}
					faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
					faces_[facenumber].b=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
					faces_[facenumber++].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
					faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
					faces_[facenumber].b=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
					faces_[facenumber++].c=vertseq[tp[3][0]][tp[3][1]][tp[3][2]];
				}
				else if((vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j+1][k].isdone)
					||( vp_[i+1][j][k].isdone && vp_[i+1][j+1][k].isdone && vp_[i][j+1][k].isdone)
					||( vp_[i+1][j+1][k].isdone && vp_[i][j+1][k].isdone && vp_[i][j][k].isdone)
					||(vp_[i][j+1][k].isdone && vp_[i][j][k].isdone && vp_[i+1][j][k].isdone))
				{
					if(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j+1][k].isdone)
					{
						tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
						tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
						tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
					}
					else if( vp_[i+1][j][k].isdone && vp_[i+1][j+1][k].isdone && vp_[i][j+1][k].isdone)
					{
						tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
						tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
						tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
					}
					else if( vp_[i+1][j+1][k].isdone && vp_[i][j+1][k].isdone && vp_[i][j][k].isdone)
					{
						tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k;
						tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
						tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
					}
					else if(vp_[i][j+1][k].isdone && vp_[i][j][k].isdone && vp_[i+1][j][k].isdone)
					{
						tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
						tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
						tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
					}
					for(ii=0;ii<3;ii++)
					{
						if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
						{
							vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
							verts_[vertnumber].x=tp[ii][0];
							verts_[vertnumber].y=tp[ii][1];
							verts_[vertnumber].z=tp[ii][2];
							vertnumber++;
						}
					}
					faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
					faces_[facenumber].b=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
					faces_[facenumber++].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
				}
				
			}
		}
	}

	///////////////////////////////////////////
	for(i=0;i<plength_-1;i++)
	{
		for(j=0;j<pwidth_-1;j++)
		{
			for(k=0;k<pheight_-1;k++)
			{
				sumtype=0;		
				for( ii=0;ii<2;ii++)
				{
					for( jj=0;jj<2;jj++)
					{
						for( kk=0;kk<2;kk++)
						{
							if(vp_[i+ii][j+jj][k+kk].isdone)
								sumtype++;
						}
					}
				}//ii
				if(vertnumber+6>allocvert)
				{
					allocvert*=2;
					verts_.resize(allocvert);
					// verts=(vertinfo *)doubleloc(verts,allocvert*sizeof(vertinfo));
				}
				if(facenumber+3>allocface)
				{
					allocface*=2;
					faces_.resize(allocface);
					// faces=(faceinfo *)doubleloc(faces,allocface*sizeof(faceinfo));
				}
				if(sumtype==0)
				{
					//nothing
				}//total0
				else if(sumtype==1)
				{
					//nothing
				}//total1
				else if(sumtype==2)
				{
					//nothing
				}//total2
				else if(sumtype==8)
				{
					//nothing
				}//total8
				
				else if(sumtype==3)
				{
					if((vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j+1][k].isdone)
					   ||(vp_[i][j][k].isdone && vp_[i][j+1][k].isdone && vp_[i+1][j+1][k].isdone)
					   ||(vp_[i][j+1][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j+1][k].isdone)
					   ||(vp_[i][j][k].isdone && vp_[i][j+1][k].isdone && vp_[i+1][j][k].isdone)
					   ||(vp_[i][j][k+1].isdone && vp_[i+1][j][k+1].isdone && vp_[i+1][j+1][k+1].isdone)
					   ||(vp_[i][j][k+1].isdone && vp_[i][j+1][k+1].isdone && vp_[i+1][j+1][k+1].isdone)
					   ||(vp_[i][j+1][k+1].isdone && vp_[i+1][j][k+1].isdone && vp_[i+1][j+1][k+1].isdone)
					   ||(vp_[i][j][k+1].isdone && vp_[i][j+1][k+1].isdone && vp_[i+1][j][k+1].isdone)
					   ||(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone)
					   ||(vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone && vp_[i][j][k+1].isdone)
					   ||(vp_[i][j][k].isdone && vp_[i+1][j][k+1].isdone && vp_[i][j][k+1].isdone)
					   ||(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i][j][k+1].isdone)
					   ||(vp_[i+1][j+1][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j+1][k+1].isdone)
					   ||(vp_[i+1][j+1][k].isdone && vp_[i+1][j][k+1].isdone && vp_[i+1][j+1][k+1].isdone)
					   ||(vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone && vp_[i+1][j+1][k+1].isdone)
					   ||(vp_[i+1][j+1][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone)
					   ||(vp_[i+1][j+1][k].isdone && vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone)
					   ||(vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone && vp_[i+1][j+1][k+1].isdone)
					   ||(vp_[i+1][j+1][k].isdone && vp_[i][j+1][k+1].isdone && vp_[i+1][j+1][k+1].isdone)
					   ||(vp_[i+1][j+1][k].isdone && vp_[i][j+1][k].isdone && vp_[i+1][j+1][k+1].isdone)
					   ||(vp_[i][j][k].isdone && vp_[i][j+1][k].isdone && vp_[i][j][k+1].isdone)
					   ||(vp_[i][j][k].isdone && vp_[i][j][k+1].isdone && vp_[i][j+1][k+1].isdone)
					   ||(vp_[i][j][k+1].isdone && vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone)
					   ||(vp_[i][j][k].isdone && vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone))
					{
						if(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j+1][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;	
						}//11
						else if(vp_[i][j][k].isdone && vp_[i][j+1][k].isdone && vp_[i+1][j+1][k].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
						}//12
						else if(vp_[i][j+1][k].isdone && vp_[i+1][j][k].isdone&& vp_[i+1][j+1][k].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
						}//13
						else if(vp_[i][j][k].isdone && vp_[i][j+1][k].isdone&& vp_[i+1][j][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
						}//14
						else if(vp_[i][j][k+1].isdone && vp_[i+1][j][k+1].isdone&& vp_[i+1][j+1][k+1].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;
						}//21
						else if(vp_[i][j][k+1].isdone && vp_[i][j+1][k+1].isdone&& vp_[i+1][j+1][k+1].isdone) 
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
						}//22
						else if(vp_[i][j+1][k+1].isdone && vp_[i+1][j][k+1].isdone&& vp_[i+1][j+1][k+1].isdone) 
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
						}//23
						else if(vp_[i][j][k+1].isdone && vp_[i][j+1][k+1].isdone&& vp_[i+1][j][k+1].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;
						}//24
						else if(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone&& vp_[i+1][j][k+1].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
						}//31
						else if(vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone && vp_[i][j][k+1].isdone) 
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
						}//32
						else if(vp_[i][j][k].isdone && vp_[i+1][j][k+1].isdone && vp_[i][j][k+1].isdone) 
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
						}//33
						else if(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i][j][k+1].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;
						}//34
						else if(vp_[i+1][j+1][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j+1][k+1].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
						}//41
						else if(vp_[i+1][j+1][k].isdone && vp_[i+1][j][k+1].isdone && vp_[i+1][j+1][k+1].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
						}//42
						else if(vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone && vp_[i+1][j+1][k+1].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
						}//43
						else if(vp_[i+1][j+1][k].isdone && vp_[i+1][j][k].isdone	&& vp_[i+1][j][k+1].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
						}//44
						else if(vp_[i+1][j+1][k].isdone && vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone ) 
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
						}//51
						else if( vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone && vp_[i+1][j+1][k+1].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
						}//52
						else if(vp_[i+1][j+1][k].isdone && vp_[i][j+1][k+1].isdone && vp_[i+1][j+1][k+1].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;
						}//53
						else if(vp_[i+1][j+1][k].isdone && vp_[i][j+1][k].isdone && vp_[i+1][j+1][k+1].isdone) 
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
						}//54
						else if(vp_[i][j][k].isdone && vp_[i][j+1][k].isdone	&& vp_[i][j][k+1].isdone ) 
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
						}//61
						else if(vp_[i][j][k].isdone 	&& vp_[i][j][k+1].isdone && vp_[i][j+1][k+1].isdone) 
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
						}//62
						else if(vp_[i][j][k+1].isdone && vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone) 
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;
						}//63
						else if(vp_[i][j][k].isdone && vp_[i][j+1][k].isdone	&& vp_[i][j+1][k+1].isdone) 
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;
						}//64
						for(ii=0;ii<3;ii++)
						{
							if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
							{
								vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
								verts_[vertnumber].x=tp[ii][0];
								verts_[vertnumber].y=tp[ii][1];
								verts_[vertnumber].z=tp[ii][2];
								vertnumber++;
							}
						}
						faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
						faces_[facenumber].b=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
						faces_[facenumber++].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
					}//no5 24
				}//total3
				else if(sumtype==4)
				{
					if((vp_[i][j][k].isdone && vp_[i+1][j][k].isdone
						&& vp_[i+1][j+1][k].isdone && vp_[i][j+1][k].isdone) 
						|| (vp_[i][j][k+1].isdone && vp_[i+1][j][k+1].isdone
						&& vp_[i+1][j+1][k+1].isdone && vp_[i][j+1][k+1].isdone)
						|| (vp_[i][j][k].isdone && vp_[i+1][j][k].isdone
						&& vp_[i+1][j][k+1].isdone && vp_[i][j][k+1].isdone)
						|| (vp_[i+1][j+1][k].isdone && vp_[i+1][j][k].isdone
						&& vp_[i+1][j][k+1].isdone && vp_[i+1][j+1][k+1].isdone)
						|| (vp_[i+1][j+1][k].isdone && vp_[i][j+1][k].isdone
						&& vp_[i][j+1][k+1].isdone && vp_[i+1][j+1][k+1].isdone)
						|| (vp_[i][j][k].isdone && vp_[i][j+1][k].isdone
						&& vp_[i][j][k+1].isdone && vp_[i][j+1][k+1].isdone))
					{
						if(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone
							&& vp_[i+1][j+1][k].isdone && vp_[i][j+1][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k;
							
						}
						else if (vp_[i][j][k+1].isdone && vp_[i+1][j][k+1].isdone
							&& vp_[i+1][j+1][k+1].isdone && vp_[i][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;
						}
						else if(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone
							&& vp_[i+1][j][k+1].isdone && vp_[i][j][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
						}
						else if(vp_[i+1][j+1][k].isdone && vp_[i+1][j][k].isdone
							&& vp_[i+1][j][k+1].isdone && vp_[i+1][j+1][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
						}
						else if(vp_[i+1][j+1][k].isdone && vp_[i][j+1][k].isdone
							&& vp_[i][j+1][k+1].isdone && vp_[i+1][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
						}
						else if(vp_[i][j][k].isdone && vp_[i][j+1][k].isdone
							&& vp_[i][j][k+1].isdone && vp_[i][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;
							tp[3][0]=i;tp[3][1]=j;tp[3][2]=k+1;
						}
						for(ii=0;ii<4;ii++)
						{
							if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
							{
								vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
								verts_[vertnumber].x=tp[ii][0];
								verts_[vertnumber].y=tp[ii][1];
								verts_[vertnumber].z=tp[ii][2];
								vertnumber++;
							}
						}
						faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
						faces_[facenumber].b=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
						faces_[facenumber++].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
						faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
						faces_[facenumber].b=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
						faces_[facenumber++].c=vertseq[tp[3][0]][tp[3][1]][tp[3][2]];
					}//no.8 6
									
				  else if((vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j+1][k].isdone  && vp_[i][j+1][k+1].isdone)//11
					   ||(vp_[i][j][k].isdone && vp_[i][j+1][k].isdone && vp_[i+1][j+1][k].isdone && vp_[i+1][j][k+1].isdone)//12
					   ||(vp_[i][j+1][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j+1][k].isdone && vp_[i][j][k+1].isdone)//13
					   ||(vp_[i][j][k].isdone && vp_[i][j+1][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j+1][k+1].isdone)//14
					   ||(vp_[i][j][k+1].isdone && vp_[i+1][j][k+1].isdone && vp_[i+1][j+1][k+1].isdone && vp_[i][j+1][k].isdone)//21
					   ||(vp_[i][j][k+1].isdone && vp_[i][j+1][k+1].isdone && vp_[i+1][j+1][k+1].isdone && vp_[i+1][j][k].isdone)//22
					   ||(vp_[i][j+1][k+1].isdone && vp_[i+1][j][k+1].isdone && vp_[i+1][j+1][k+1].isdone && vp_[i][j][k].isdone)//23
					   ||(vp_[i][j][k+1].isdone && vp_[i][j+1][k+1].isdone && vp_[i+1][j][k+1].isdone && vp_[i+1][j+1][k].isdone)//24
					   ||(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone && vp_[i][j+1][k+1].isdone)//31
					   ||(vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone && vp_[i][j][k+1].isdone && vp_[i][j+1][k].isdone)//32
					   ||(vp_[i][j][k].isdone && vp_[i+1][j][k+1].isdone && vp_[i][j][k+1].isdone && vp_[i+1][j+1][k].isdone)//33
					   ||(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i][j][k+1].isdone && vp_[i+1][j+1][k+1].isdone)//34
					   ||(vp_[i+1][j+1][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j+1][k+1].isdone && vp_[i][j][k+1].isdone)//41
					   ||(vp_[i+1][j+1][k].isdone && vp_[i+1][j][k+1].isdone && vp_[i+1][j+1][k+1].isdone && vp_[i][j][k].isdone)//42
					   ||(vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone && vp_[i+1][j+1][k+1].isdone && vp_[i][j+1][k].isdone)//43
					   ||(vp_[i+1][j+1][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone  && vp_[i][j+1][k+1].isdone)//44
					   ||(vp_[i+1][j+1][k].isdone && vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone  && vp_[i+1][j][k+1].isdone)//51
					   ||( vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone && vp_[i+1][j+1][k+1].isdone && vp_[i+1][j][k].isdone)//52
					   ||(vp_[i+1][j+1][k].isdone && vp_[i][j+1][k+1].isdone && vp_[i+1][j+1][k+1].isdone && vp_[i][j][k].isdone)//53
					   ||(vp_[i+1][j+1][k].isdone && vp_[i][j+1][k].isdone && vp_[i+1][j+1][k+1].isdone && vp_[i][j][k+1].isdone)//54
					   ||(vp_[i][j][k].isdone && vp_[i][j+1][k].isdone && vp_[i][j][k+1].isdone  && vp_[i+1][j+1][k+1].isdone)//61
					   ||(vp_[i][j][k].isdone && vp_[i][j][k+1].isdone && vp_[i][j+1][k+1].isdone && vp_[i+1][j+1][k].isdone)//62
					   ||(vp_[i][j][k+1].isdone && vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone && vp_[i+1][j][k].isdone)//63
					   ||(vp_[i][j][k].isdone && vp_[i][j+1][k].isdone&& vp_[i][j+1][k+1].isdone && vp_[i+1][j][k+1].isdone))
				   {
						if(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j+1][k].isdone  && vp_[i][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;	
						}//11
						else if(vp_[i][j][k].isdone && vp_[i][j+1][k].isdone && vp_[i+1][j+1][k].isdone && vp_[i+1][j][k+1].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
						}//12
						else if(vp_[i][j+1][k].isdone && vp_[i+1][j][k].isdone&& vp_[i+1][j+1][k].isdone && vp_[i][j][k+1].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
						}//13
						else if(vp_[i][j][k].isdone && vp_[i][j+1][k].isdone&& vp_[i+1][j][k].isdone && vp_[i+1][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
						}//14
						else if(vp_[i][j][k+1].isdone && vp_[i+1][j][k+1].isdone&& vp_[i+1][j+1][k+1].isdone && vp_[i][j+1][k].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;
						}//21
						else if(vp_[i][j][k+1].isdone && vp_[i][j+1][k+1].isdone&& vp_[i+1][j+1][k+1].isdone && vp_[i+1][j][k].isdone) 
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
						}//22
						else if(vp_[i][j+1][k+1].isdone && vp_[i+1][j][k+1].isdone&& vp_[i+1][j+1][k+1].isdone && vp_[i][j][k].isdone) 
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
						}//23
						else if(vp_[i][j][k+1].isdone && vp_[i][j+1][k+1].isdone&& vp_[i+1][j][k+1].isdone && vp_[i+1][j+1][k].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;
						}//24
						else if(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone&& vp_[i+1][j][k+1].isdone && vp_[i][j+1][k+1].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
						}//31
						else if(vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone && vp_[i][j][k+1].isdone && vp_[i][j+1][k].isdone) 
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
						}//32
						else if(vp_[i][j][k].isdone && vp_[i+1][j][k+1].isdone && vp_[i][j][k+1].isdone && vp_[i+1][j+1][k].isdone) 
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
						}//33
						else if(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone && vp_[i][j][k+1].isdone && vp_[i+1][j+1][k+1].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;
						}//34
						else if(vp_[i+1][j+1][k].isdone && vp_[i+1][j][k].isdone && vp_[i+1][j+1][k+1].isdone && vp_[i][j][k+1].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
						}//41
						else if(vp_[i+1][j+1][k].isdone && vp_[i+1][j][k+1].isdone && vp_[i+1][j+1][k+1].isdone && vp_[i][j][k].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
						}//42
						else if(vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone && vp_[i+1][j+1][k+1].isdone && vp_[i][j+1][k].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
						}//43
						else if(vp_[i+1][j+1][k].isdone && vp_[i+1][j][k].isdone	&& vp_[i+1][j][k+1].isdone && vp_[i][j+1][k+1].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
						}//44
						else if(vp_[i+1][j+1][k].isdone && vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone && vp_[i+1][j][k+1].isdone ) 
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
						}//51
						else if( vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone && vp_[i+1][j+1][k+1].isdone && vp_[i+1][j][k].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
						}//52
						else if(vp_[i+1][j+1][k].isdone && vp_[i][j+1][k+1].isdone && vp_[i+1][j+1][k+1].isdone && vp_[i][j][k].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;
						}//53
						else if(vp_[i+1][j+1][k].isdone && vp_[i][j+1][k].isdone && vp_[i+1][j+1][k+1].isdone && vp_[i][j][k+1].isdone) 
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
						}//54
						else if(vp_[i][j][k].isdone && vp_[i][j+1][k].isdone	&& vp_[i][j][k+1].isdone && vp_[i+1][j+1][k+1].isdone) 
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
						}//61
						else if(vp_[i][j][k].isdone 	&& vp_[i][j][k+1].isdone && vp_[i][j+1][k+1].isdone && vp_[i+1][j+1][k].isdone) 
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
						}//62
						else if(vp_[i][j][k+1].isdone && vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone && vp_[i+1][j][k].isdone) 
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;
						}//63
						else if(vp_[i][j][k].isdone && vp_[i][j+1][k].isdone	&& vp_[i][j+1][k+1].isdone && vp_[i+1][j][k+1].isdone) 
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;
						}//64
						for(ii=0;ii<3;ii++)
						{
							if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
							{
								vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
								verts_[vertnumber].x=tp[ii][0];
								verts_[vertnumber].y=tp[ii][1];
								verts_[vertnumber].z=tp[ii][2];
								vertnumber++;
							}
						}
						faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
						faces_[facenumber].b=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
						faces_[facenumber++].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
				   }//no12 24
					else if((vp_[i][j][k].isdone && vp_[i][j+1][k+1].isdone
						&& vp_[i+1][j+1][k].isdone && vp_[i][j+1][k].isdone)
						|| (vp_[i][j][k].isdone && vp_[i+1][j][k].isdone
						&& vp_[i+1][j+1][k].isdone && vp_[i+1][j][k+1].isdone)
						|| (vp_[i][j][k].isdone && vp_[i][j][k+1].isdone
						&& vp_[i+1][j][k].isdone && vp_[i][j+1][k].isdone)
						|| (vp_[i][j+1][k].isdone && vp_[i+1][j][k].isdone
						&& vp_[i+1][j+1][k].isdone && vp_[i+1][j+1][k+1].isdone)
						|| (vp_[i][j][k+1].isdone && vp_[i][j+1][k+1].isdone
						&& vp_[i+1][j+1][k+1].isdone && vp_[i][j+1][k].isdone)
						|| (vp_[i][j][k+1].isdone && vp_[i+1][j][k].isdone
						&& vp_[i+1][j+1][k+1].isdone && vp_[i+1][j][k+1].isdone)
						|| (vp_[i][j][k].isdone && vp_[i][j][k+1].isdone
						&& vp_[i+1][j][k+1].isdone && vp_[i][j+1][k+1].isdone)
						|| (vp_[i][j+1][k+1].isdone && vp_[i+1][j][k+1].isdone
						&& vp_[i+1][j+1][k].isdone && vp_[i+1][j+1][k+1].isdone))
					{
						if(vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone
							&& vp_[i][j][k].isdone && vp_[i+1][j+1][k].isdone )
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
						}//1
						else if(vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone
							&& vp_[i+1][j+1][k].isdone && vp_[i][j][k].isdone )
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
						}//2
						else if(vp_[i][j][k].isdone && vp_[i][j][k+1].isdone
							&& vp_[i+1][j][k].isdone && vp_[i][j+1][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
						}//3
						else if(vp_[i+1][j+1][k].isdone && vp_[i+1][j+1][k+1].isdone
							&& vp_[i][j+1][k].isdone && vp_[i+1][j][k].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
						}//4
						else if(vp_[i][j+1][k].isdone && vp_[i][j+1][k+1].isdone
							&& vp_[i+1][j+1][k+1].isdone && vp_[i][j][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k+1;
						}//5
						else if(vp_[i+1][j][k].isdone && vp_[i+1][j][k+1].isdone
							&& vp_[i+1][j+1][k+1].isdone && vp_[i][j][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
						}//6
						else if(vp_[i][j][k].isdone && vp_[i][j][k+1].isdone
							&& vp_[i+1][j][k+1].isdone && vp_[i][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;
						}//7
						else if(vp_[i][j+1][k+1].isdone && vp_[i+1][j][k+1].isdone
							&& vp_[i+1][j+1][k].isdone && vp_[i+1][j+1][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
						}//8
						for(ii=0;ii<3;ii++)
						{
							if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
							{
								vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
								verts_[vertnumber].x=tp[ii][0];
								verts_[vertnumber].y=tp[ii][1];
								verts_[vertnumber].z=tp[ii][2];
								vertnumber++;
							}
						}
						faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
						faces_[facenumber].b=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
						faces_[facenumber++].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
					}// no.9 8
					else if((vp_[i][j][k].isdone && vp_[i+1][j][k].isdone
						&& vp_[i+1][j+1][k].isdone && vp_[i][j][k+1].isdone)
						||(vp_[i][j+1][k].isdone && vp_[i+1][j][k].isdone
						&& vp_[i+1][j+1][k].isdone && vp_[i+1][j][k+1].isdone)
						||(vp_[i][j+1][k].isdone && vp_[i][j][k].isdone
						&& vp_[i+1][j+1][k].isdone && vp_[i+1][j+1][k+1].isdone)
						||(vp_[i][j+1][k].isdone && vp_[i][j][k].isdone
						&& vp_[i+1][j][k].isdone && vp_[i][j+1][k+1].isdone)
						||(vp_[i][j+1][k+1].isdone && vp_[i][j][k+1].isdone
						&& vp_[i+1][j][k+1].isdone && vp_[i+1][j][k].isdone)
						||(vp_[i+1][j+1][k+1].isdone && vp_[i][j][k+1].isdone
						&& vp_[i+1][j][k+1].isdone && vp_[i+1][j+1][k].isdone)
						||(vp_[i+1][j+1][k+1].isdone && vp_[i][j+1][k+1].isdone
						&& vp_[i+1][j][k+1].isdone && vp_[i][j+1][k].isdone)
						||(vp_[i+1][j+1][k+1].isdone && vp_[i][j+1][k+1].isdone
						&& vp_[i][j][k+1].isdone && vp_[i][j][k].isdone)
						||(vp_[i+1][j+1][k].isdone && vp_[i][j+1][k+1].isdone
						&& vp_[i][j][k+1].isdone && vp_[i][j+1][k].isdone)
						||(vp_[i+1][j][k+1].isdone && vp_[i][j][k].isdone
						&& vp_[i][j][k+1].isdone && vp_[i][j+1][k].isdone)
						||(vp_[i+1][j][k+1].isdone && vp_[i][j][k].isdone
						&& vp_[i+1][j+1][k+1].isdone && vp_[i+1][j][k].isdone)
						||(vp_[i][j+1][k+1].isdone && vp_[i+1][j+1][k].isdone
						&& vp_[i+1][j+1][k+1].isdone && vp_[i+1][j][k].isdone))
					{
						if(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone
							&& vp_[i+1][j+1][k].isdone && vp_[i][j][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;	
							tp[3][0]=i;tp[3][1]=j;tp[3][2]=k+1;
						}//1
						else if(vp_[i][j+1][k].isdone && vp_[i+1][j][k].isdone
							&& vp_[i+1][j+1][k].isdone && vp_[i+1][j][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;	
							tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k+1;
						}//2
						else if(vp_[i][j+1][k].isdone && vp_[i][j][k].isdone
							&& vp_[i+1][j+1][k].isdone && vp_[i+1][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;	
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k+1;
						}//3
						else if(vp_[i][j+1][k].isdone && vp_[i][j][k].isdone
							&& vp_[i+1][j][k].isdone && vp_[i][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;	
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
						}//4
						else if(vp_[i][j+1][k+1].isdone && vp_[i][j][k+1].isdone
							&& vp_[i+1][j][k+1].isdone && vp_[i+1][j][k].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;	
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
						}//5
						else if(vp_[i+1][j+1][k+1].isdone && vp_[i][j][k+1].isdone
							&& vp_[i+1][j][k+1].isdone && vp_[i+1][j+1][k].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;	
							tp[3][0]=i;tp[3][1]=j;tp[3][2]=k+1;
						}//6
						else if(vp_[i+1][j+1][k+1].isdone && vp_[i][j+1][k+1].isdone
							&& vp_[i+1][j][k+1].isdone && vp_[i][j+1][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;	
							tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k+1;
						}//7
						else if(vp_[i+1][j+1][k+1].isdone && vp_[i][j+1][k+1].isdone
							&& vp_[i][j][k+1].isdone && vp_[i][j][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;	
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k+1;
						}//8
						else if(vp_[i+1][j+1][k].isdone && vp_[i][j+1][k+1].isdone
							&& vp_[i][j][k+1].isdone && vp_[i][j+1][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;	
							tp[3][0]=i;tp[3][1]=j;tp[3][2]=k+1;
						}//9
						else if(vp_[i+1][j][k+1].isdone && vp_[i][j][k].isdone
							&& vp_[i][j][k+1].isdone && vp_[i][j+1][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;	
							tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k+1;
						}//10
						else if(vp_[i+1][j][k+1].isdone && vp_[i][j][k].isdone
							&& vp_[i+1][j+1][k+1].isdone && vp_[i+1][j][k].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;	
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k+1;
						}//11
						else if(vp_[i][j+1][k+1].isdone && vp_[i+1][j+1][k].isdone
							&& vp_[i+1][j+1][k+1].isdone && vp_[i+1][j][k].isdone) 
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;	
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
						}//12
						for(ii=0;ii<4;ii++)
						{
							if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
							{
								vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
								verts_[vertnumber].x=tp[ii][0];
								verts_[vertnumber].y=tp[ii][1];
								verts_[vertnumber].z=tp[ii][2];
								vertnumber++;
							}
						}
						faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
						faces_[facenumber].b=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
						faces_[facenumber++].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
						faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
						faces_[facenumber].b=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
						faces_[facenumber++].c=vertseq[tp[3][0]][tp[3][1]][tp[3][2]];
					}//no.11 12
					else if((vp_[i][j][k].isdone && vp_[i+1][j][k].isdone
						&& vp_[i][j+1][k].isdone && vp_[i+1][j][k+1].isdone)
						||(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone
						&& vp_[i+1][j+1][k].isdone && vp_[i+1][j+1][k+1].isdone)
						||(vp_[i][j+1][k].isdone && vp_[i+1][j][k].isdone
						&& vp_[i+1][j+1][k].isdone && vp_[i][j+1][k+1].isdone)
						||(vp_[i][j+1][k].isdone && vp_[i][j][k].isdone
						&& vp_[i+1][j+1][k].isdone && vp_[i][j][k+1].isdone)
						||(vp_[i+1][j+1][k+1].isdone && vp_[i][j][k+1].isdone
						&& vp_[i+1][j][k+1].isdone && vp_[i][j][k].isdone)
						||(vp_[i+1][j+1][k+1].isdone && vp_[i][j+1][k+1].isdone
						&& vp_[i+1][j][k+1].isdone && vp_[i+1][j][k].isdone)
						||(vp_[i+1][j+1][k+1].isdone && vp_[i][j+1][k+1].isdone
						&& vp_[i][j][k+1].isdone && vp_[i+1][j+1][k].isdone)
						||(vp_[i+1][j][k+1].isdone && vp_[i][j+1][k+1].isdone
						&& vp_[i][j][k+1].isdone && vp_[i][j+1][k].isdone)
						||(vp_[i+1][j+1][k+1].isdone && vp_[i][j+1][k+1].isdone
						&& vp_[i][j][k].isdone && vp_[i][j+1][k].isdone)
						||(vp_[i+1][j][k].isdone && vp_[i][j][k].isdone
						&& vp_[i][j][k+1].isdone && vp_[i][j+1][k+1].isdone)
						||(vp_[i+1][j][k+1].isdone && vp_[i][j][k+1].isdone
						&& vp_[i+1][j+1][k].isdone && vp_[i+1][j][k].isdone)
						||(vp_[i][j+1][k].isdone && vp_[i+1][j+1][k].isdone
						&& vp_[i+1][j+1][k+1].isdone && vp_[i+1][j][k+1].isdone))
					{
						if(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone
							&& vp_[i][j+1][k].isdone && vp_[i+1][j][k+1].isdone)  
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;	
							tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k+1;
						}//1
						else if(vp_[i][j][k].isdone && vp_[i+1][j][k].isdone
							&& vp_[i+1][j+1][k].isdone && vp_[i+1][j+1][k+1].isdone)  
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;	
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k+1;
						}//2
						else if(vp_[i][j+1][k].isdone && vp_[i+1][j][k].isdone
							&& vp_[i+1][j+1][k].isdone && vp_[i][j+1][k+1].isdone)  
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;	
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
						}//3
						else if(vp_[i][j+1][k].isdone && vp_[i][j][k].isdone
							&& vp_[i+1][j+1][k].isdone && vp_[i][j][k+1].isdone)  
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;	
							tp[3][0]=i;tp[3][1]=j;tp[3][2]=k+1;
						}//4
						else if(vp_[i+1][j+1][k+1].isdone && vp_[i][j][k+1].isdone
							&& vp_[i+1][j][k+1].isdone && vp_[i][j][k].isdone)  
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;	
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k+1;
						}//5
						else if(vp_[i+1][j+1][k+1].isdone && vp_[i][j+1][k+1].isdone
							&& vp_[i+1][j][k+1].isdone && vp_[i+1][j][k].isdone)  
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;	
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
						}//6
						else if(vp_[i+1][j+1][k+1].isdone && vp_[i][j+1][k+1].isdone
							&& vp_[i][j][k+1].isdone && vp_[i+1][j+1][k].isdone)  
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;	
							tp[3][0]=i;tp[3][1]=j;tp[3][2]=k+1;
						}//7
						else if(vp_[i+1][j][k+1].isdone && vp_[i][j+1][k+1].isdone
							&& vp_[i][j][k+1].isdone && vp_[i][j+1][k].isdone)  
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;	
							tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k+1;
						}//8
						else if(vp_[i+1][j+1][k+1].isdone && vp_[i][j+1][k+1].isdone
							&& vp_[i][j][k].isdone && vp_[i][j+1][k].isdone)  
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;	
							tp[3][0]=i;tp[3][1]=j;tp[3][2]=k;
						}//9
						else if(vp_[i+1][j][k].isdone && vp_[i][j][k].isdone
							&& vp_[i][j][k+1].isdone && vp_[i][j+1][k+1].isdone)  
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;	
							tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k;
						}//10
						else if(vp_[i+1][j][k+1].isdone && vp_[i][j][k+1].isdone
							&& vp_[i+1][j+1][k].isdone && vp_[i+1][j][k].isdone)  
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;	
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k;
						}//11
						else if(vp_[i][j+1][k].isdone && vp_[i+1][j+1][k].isdone
							&& vp_[i+1][j+1][k+1].isdone && vp_[i+1][j][k+1].isdone)  
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;	
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k;
						}//12
						for(ii=0;ii<4;ii++)
						{
							if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
							{
								vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
								verts_[vertnumber].x=tp[ii][0];
								verts_[vertnumber].y=tp[ii][1];
								verts_[vertnumber].z=tp[ii][2];
								vertnumber++;
							}
						}
						faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
						faces_[facenumber].b=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
						faces_[facenumber++].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
						faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
						faces_[facenumber].b=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
						faces_[facenumber++].c=vertseq[tp[3][0]][tp[3][1]][tp[3][2]];
					}//no.14 12
				}//total4
				else if(sumtype==5)
				{
					if((!vp_[i+1][j][k].isdone && !vp_[i][j][k+1].isdone && !vp_[i+1][j+1][k+1].isdone)
						|| (!vp_[i][j+1][k].isdone && !vp_[i][j][k+1].isdone && !vp_[i+1][j+1][k+1].isdone)
						|| (!vp_[i+1][j+1][k].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i][j+1][k+1].isdone)
						|| (!vp_[i][j][k].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i][j+1][k+1].isdone)
						|| (!vp_[i+1][j][k+1].isdone && !vp_[i][j][k].isdone && !vp_[i+1][j+1][k].isdone)
						|| (!vp_[i][j+1][k+1].isdone && !vp_[i][j][k].isdone && !vp_[i+1][j+1][k].isdone)
						|| (!vp_[i+1][j+1][k+1].isdone && !vp_[i+1][j][k].isdone && !vp_[i][j+1][k].isdone)
						|| (!vp_[i][j][k+1].isdone && !vp_[i+1][j][k].isdone && !vp_[i][j+1][k].isdone))
					{		
						if(!vp_[i+1][j][k].isdone && !vp_[i][j][k+1].isdone && !vp_[i+1][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
						}//1
						else if(!vp_[i][j+1][k].isdone && !vp_[i][j][k+1].isdone && !vp_[i+1][j+1][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
						}//2
						else if(!vp_[i+1][j+1][k].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
						}//3
						else if(!vp_[i][j][k].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i][j+1][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
						}//4
						else if(!vp_[i+1][j][k+1].isdone && !vp_[i][j][k].isdone && !vp_[i+1][j+1][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k+1;
						}//5
						else if(!vp_[i][j+1][k+1].isdone && !vp_[i][j][k].isdone && !vp_[i+1][j+1][k].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
						}//6
						else if(!vp_[i+1][j+1][k+1].isdone && !vp_[i+1][j][k].isdone && !vp_[i][j+1][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;
						}//7
						else if(!vp_[i][j][k+1].isdone && !vp_[i+1][j][k].isdone && !vp_[i][j+1][k].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
						}//8
						for(ii=0;ii<3;ii++)
						{
							if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
							{
								vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
								verts_[vertnumber].x=tp[ii][0];
								verts_[vertnumber].y=tp[ii][1];
								verts_[vertnumber].z=tp[ii][2];
								vertnumber++;
							}
						}
						faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
						faces_[facenumber].b=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
						faces_[facenumber++].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
					}//no.7 8
					else if((!vp_[i][j][k].isdone && !vp_[i+1][j][k].isdone && !vp_[i+1][j+1][k].isdone)
				   ||(!vp_[i][j][k].isdone && !vp_[i][j+1][k].isdone && !vp_[i+1][j+1][k].isdone)
				   ||(!vp_[i][j+1][k].isdone && !vp_[i+1][j][k].isdone && !vp_[i+1][j+1][k].isdone)
				   ||(!vp_[i][j][k].isdone && !vp_[i][j+1][k].isdone && !vp_[i+1][j][k].isdone)
				   ||(!vp_[i][j][k+1].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i+1][j+1][k+1].isdone)
				   ||(!vp_[i][j][k+1].isdone && !vp_[i][j+1][k+1].isdone && !vp_[i+1][j+1][k+1].isdone)
				   ||(!vp_[i][j+1][k+1].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i+1][j+1][k+1].isdone)
				   ||(!vp_[i][j][k+1].isdone && !vp_[i][j+1][k+1].isdone && !vp_[i+1][j][k+1].isdone)
				   ||(!vp_[i][j][k].isdone && !vp_[i+1][j][k].isdone && !vp_[i+1][j][k+1].isdone)
				   ||(!vp_[i+1][j][k].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i][j][k+1].isdone)
				   ||(!vp_[i][j][k].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i][j][k+1].isdone)
				   ||(!vp_[i][j][k].isdone && !vp_[i+1][j][k].isdone && !vp_[i][j][k+1].isdone)
				   ||(!vp_[i+1][j+1][k].isdone && !vp_[i+1][j][k].isdone && !vp_[i+1][j+1][k+1].isdone)
				   ||(!vp_[i+1][j+1][k].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i+1][j+1][k+1].isdone)
				   ||(!vp_[i+1][j][k].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i+1][j+1][k+1].isdone)
				   ||(!vp_[i+1][j+1][k].isdone && !vp_[i+1][j][k].isdone && !vp_[i+1][j][k+1].isdone )
				   ||(!vp_[i+1][j+1][k].isdone && !vp_[i][j+1][k].isdone && !vp_[i][j+1][k+1].isdone )
				   ||(!vp_[i][j+1][k].isdone && !vp_[i][j+1][k+1].isdone && !vp_[i+1][j+1][k+1].isdone)
				   ||(!vp_[i+1][j+1][k].isdone && !vp_[i][j+1][k+1].isdone && !vp_[i+1][j+1][k+1].isdone)
				   ||(!vp_[i+1][j+1][k].isdone && !vp_[i][j+1][k].isdone && !vp_[i+1][j+1][k+1].isdone)
				   ||(!vp_[i][j][k].isdone && !vp_[i][j+1][k].isdone && !vp_[i][j][k+1].isdone )
				   ||(!vp_[i][j][k].isdone && !vp_[i][j][k+1].isdone && !vp_[i][j+1][k+1].isdone)
				   ||(!vp_[i][j][k+1].isdone && !vp_[i][j+1][k].isdone && !vp_[i][j+1][k+1].isdone)
				   ||(!vp_[i][j][k].isdone && !vp_[i][j+1][k].isdone && !vp_[i][j+1][k+1].isdone))
				{
					if(!vp_[i][j][k].isdone && !vp_[i+1][j][k].isdone && !vp_[i+1][j+1][k].isdone)
					{
						tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;
						tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
						tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
						tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k;
					}//11
					else if(!vp_[i][j][k].isdone && !vp_[i][j+1][k].isdone && !vp_[i+1][j+1][k].isdone) 
					{
						tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
						tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;
						tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
						tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k;
					}//12
					else if(!vp_[i][j+1][k].isdone && !vp_[i+1][j][k].isdone&& !vp_[i+1][j+1][k].isdone) 
					{
						tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
						tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k+1;
						tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
						tp[3][0]=i;tp[3][1]=j;tp[3][2]=k;
					}//13
					else if(!vp_[i][j][k].isdone && !vp_[i][j+1][k].isdone&& !vp_[i+1][j][k].isdone)
					{
						tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;
						tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
						tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
						tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k;
					}//14
					else if(!vp_[i][j][k+1].isdone && !vp_[i+1][j][k+1].isdone&& !vp_[i+1][j+1][k+1].isdone) 
					{
						tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
						tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
						tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
						tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
					}//21
					else if(!vp_[i][j][k+1].isdone && !vp_[i][j+1][k+1].isdone&& !vp_[i+1][j+1][k+1].isdone) 
					{
						tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
						tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
						tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k;
						tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k+1;
					}//22
					else if(!vp_[i][j+1][k+1].isdone && !vp_[i+1][j][k+1].isdone&& !vp_[i+1][j+1][k+1].isdone) 
					{
						tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
						tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
						tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
						tp[3][0]=i;tp[3][1]=j;tp[3][2]=k+1;
					}//23
					else if(!vp_[i][j][k+1].isdone && !vp_[i][j+1][k+1].isdone&& !vp_[i+1][j][k+1].isdone) 
					{
						tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
						tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
						tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
						tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k+1;
					}//24
					else if(!vp_[i][j][k].isdone && !vp_[i+1][j][k].isdone&& !vp_[i+1][j][k+1].isdone) 
					{
						tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
						tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
						tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
						tp[3][0]=i;tp[3][1]=j;tp[3][2]=k+1;
					}//31
					else if(!vp_[i+1][j][k].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i][j][k+1].isdone) 
					{
						tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;
						tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k+1;
						tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k;
						tp[3][0]=i;tp[3][1]=j;tp[3][2]=k;
					}//32
					else if(!vp_[i][j][k].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i][j][k+1].isdone) 
					{
						tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
						tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;
						tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
						tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k;
					}//33
					else if(!vp_[i][j][k].isdone && !vp_[i+1][j][k].isdone && !vp_[i][j][k+1].isdone) 
					{
						tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
						tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
						tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
						tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k+1;
					}//34
					else if(!vp_[i+1][j+1][k].isdone && !vp_[i+1][j][k].isdone && !vp_[i+1][j+1][k+1].isdone) 
					{
						tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;
						tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
						tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
						tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k+1;
					}//41
					else if(!vp_[i+1][j+1][k].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i+1][j+1][k+1].isdone) 
					{
						tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;
						tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;
						tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
						tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k;
					}//42
					else if(!vp_[i+1][j][k].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i+1][j+1][k+1].isdone) 
					{
						tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
						tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
						tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
						tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k;
					}//43
					else if(!vp_[i+1][j+1][k].isdone && !vp_[i+1][j][k].isdone && !vp_[i+1][j][k+1].isdone) 
					{
						tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
						tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
						tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
						tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k+1;
					}//44
					else if(!vp_[i+1][j+1][k].isdone && !vp_[i][j+1][k].isdone && !vp_[i][j+1][k+1].isdone ) 
					{
						tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;
						tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
						tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
						tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k+1;
					}//51
					else if( !vp_[i][j+1][k].isdone && !vp_[i][j+1][k+1].isdone && !vp_[i+1][j+1][k+1].isdone) 
					{
						tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
						tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
						tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
						tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k;
					}//52
					else if(!vp_[i+1][j+1][k].isdone && !vp_[i][j+1][k+1].isdone && !vp_[i+1][j+1][k+1].isdone) 
					{
						tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
						tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
						tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
						tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k;
					}//53
					else if(!vp_[i+1][j+1][k].isdone && !vp_[i][j+1][k].isdone && !vp_[i+1][j+1][k+1].isdone) 
					{
						tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
						tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
						tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
						tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
					}//54
					else if(!vp_[i][j][k].isdone && !vp_[i][j+1][k].isdone && !vp_[i][j][k+1].isdone ) 
					{
						tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
						tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
						tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k;
						tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
					}//61
					else if(!vp_[i][j][k].isdone && !vp_[i][j][k+1].isdone && !vp_[i][j+1][k+1].isdone) 
					{
						tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
						tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
						tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
						tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k;
					}//62
					else if(!vp_[i][j][k+1].isdone && !vp_[i][j+1][k].isdone && !vp_[i][j+1][k+1].isdone) 
					{
						tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
						tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k+1;
						tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
						tp[3][0]=i;tp[3][1]=j;tp[3][2]=k;
					}//63
					else if(!vp_[i][j][k].isdone && !vp_[i][j+1][k].isdone && !vp_[i][j+1][k+1].isdone) 
					{
						tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
						tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
						tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
						tp[3][0]=i;tp[3][1]=j;tp[3][2]=k+1;
					}//64
					for(ii=0;ii<4;ii++)
					{
						if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
						{
							vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
							verts_[vertnumber].x=tp[ii][0];
							verts_[vertnumber].y=tp[ii][1];
							verts_[vertnumber].z=tp[ii][2];
							vertnumber++;
						}
					}
					faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
					faces_[facenumber].b=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
					faces_[facenumber++].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
					faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
					faces_[facenumber].b=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
					faces_[facenumber++].c=vertseq[tp[3][0]][tp[3][1]][tp[3][2]];
				}//no5 24
					else if((!vp_[i][j][k].isdone && !vp_[i+1][j][k].isdone && !vp_[i+1][j+1][k+1].isdone)//1
						||(!vp_[i][j+1][k].isdone && !vp_[i+1][j+1][k].isdone && !vp_[i][j][k+1].isdone)//2
						||(!vp_[i][j+1][k+1].isdone && !vp_[i+1][j+1][k+1].isdone && !vp_[i+1][j][k].isdone)//3
						||(!vp_[i][j][k+1].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i+1][j+1][k].isdone)//4
						||(!vp_[i][j][k].isdone && !vp_[i][j+1][k].isdone && !vp_[i+1][j+1][k+1].isdone)//5
						||(!vp_[i+1][j][k+1].isdone && !vp_[i+1][j+1][k+1].isdone && !vp_[i][j+1][k].isdone)//6
						||(!vp_[i+1][j][k].isdone && !vp_[i+1][j+1][k].isdone && !vp_[i][j+1][k+1].isdone)//7
						||(!vp_[i][j][k+1].isdone && !vp_[i][j+1][k+1].isdone && !vp_[i+1][j+1][k].isdone)//8
						||(!vp_[i][j][k].isdone && !vp_[i][j][k+1].isdone && !vp_[i+1][j+1][k+1].isdone)//9
						||(!vp_[i+1][j+1][k].isdone && !vp_[i+1][j+1][k+1].isdone && !vp_[i][j][k].isdone)//10
						||(!vp_[i+1][j][k].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i][j+1][k+1].isdone)//11
						||(!vp_[i][j+1][k].isdone && !vp_[i][j+1][k+1].isdone && !vp_[i+1][j][k+1].isdone))
					{
						if(!vp_[i][j][k].isdone && !vp_[i+1][j][k].isdone && !vp_[i+1][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
							tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k+1;
							tp[4][0]=i;tp[4][1]=j+1;tp[4][2]=k+1;
						}//1
						else if(!vp_[i][j+1][k].isdone && !vp_[i+1][j+1][k].isdone && !vp_[i][j][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
							tp[4][0]=i+1;tp[4][1]=j;tp[4][2]=k+1;
						}//2
						else if(!vp_[i][j+1][k+1].isdone && !vp_[i+1][j+1][k+1].isdone && !vp_[i+1][j][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k;
							tp[4][0]=i;tp[4][1]=j;tp[4][2]=k;
						}//3
						else if(!vp_[i][j][k+1].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i+1][j+1][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k+1;
							tp[4][0]=i;tp[4][1]=j+1;tp[4][2]=k;
						}//4
						else if(!vp_[i][j][k].isdone && !vp_[i][j+1][k].isdone && !vp_[i+1][j+1][k+1].isdone)
						{
						    //tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							//tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							//tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
							//tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;
							tp[4][0]=i+1;tp[4][1]=j;tp[4][2]=k+1;
						}//5
						else if(!vp_[i+1][j][k+1].isdone && !vp_[i+1][j+1][k+1].isdone && !vp_[i][j+1][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
							tp[4][0]=i;tp[4][1]=j;tp[4][2]=k;
						}//6
						else if(!vp_[i+1][j][k].isdone && !vp_[i+1][j+1][k].isdone && !vp_[i][j+1][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k+1;
							tp[4][0]=i;tp[4][1]=j;tp[4][2]=k+1;
						}//7
						else if(!vp_[i][j][k+1].isdone && !vp_[i][j+1][k+1].isdone && !vp_[i+1][j+1][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k;
							tp[4][0]=i+1;tp[4][1]=j;tp[4][2]=k;
						}//8
						else if(!vp_[i][j][k].isdone && !vp_[i][j][k+1].isdone && !vp_[i+1][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
							tp[4][0]=i+1;tp[4][1]=j+1;tp[4][2]=k;
						}//9
						else if(!vp_[i+1][j+1][k].isdone && !vp_[i+1][j+1][k+1].isdone && !vp_[i][j][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k;
							tp[4][0]=i;tp[4][1]=j;tp[4][2]=k+1;
						}//10
						else if(!vp_[i+1][j][k].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
							tp[3][0]=i;tp[3][1]=j;tp[3][2]=k+1;
							tp[4][0]=i;tp[4][1]=j+1;tp[4][2]=k;
						}//11
						else if(!vp_[i][j+1][k].isdone && !vp_[i][j+1][k+1].isdone && !vp_[i+1][j][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k+1;
							tp[4][0]=i+1;tp[4][1]=j;tp[4][2]=k;
						}//12
						for(ii=0;ii<5;ii++)
						{
							if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
							{
								vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
								verts_[vertnumber].x=tp[ii][0];
								verts_[vertnumber].y=tp[ii][1];
								verts_[vertnumber].z=tp[ii][2];
								vertnumber++;
							}
						}
						faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
						faces_[facenumber].b=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
						faces_[facenumber++].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
						faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
						faces_[facenumber].b=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
						faces_[facenumber++].c=vertseq[tp[3][0]][tp[3][1]][tp[3][2]];
						faces_[facenumber].a=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
						faces_[facenumber].b=vertseq[tp[4][0]][tp[4][1]][tp[4][2]];
						faces_[facenumber++].c=vertseq[tp[3][0]][tp[3][1]][tp[3][2]];
						
					}//no.6 12-1
					else if((!vp_[i][j][k].isdone && !vp_[i+1][j][k].isdone && !vp_[i][j+1][k+1].isdone)//1
						||(!vp_[i][j+1][k].isdone && !vp_[i+1][j+1][k].isdone && !vp_[i+1][j][k+1].isdone)//2
						||(!vp_[i][j+1][k+1].isdone && !vp_[i+1][j+1][k+1].isdone && !vp_[i][j][k].isdone)//3
						||(!vp_[i][j][k+1].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i][j+1][k].isdone)//4
						||(!vp_[i][j][k].isdone && !vp_[i][j+1][k].isdone && !vp_[i+1][j][k+1].isdone)//5
						||(!vp_[i+1][j][k+1].isdone && !vp_[i+1][j+1][k+1].isdone && !vp_[i][j][k].isdone)//6
						||(!vp_[i+1][j][k].isdone && !vp_[i+1][j+1][k].isdone && !vp_[i][j][k+1].isdone)//7
						||(!vp_[i][j][k+1].isdone && !vp_[i][j+1][k+1].isdone && !vp_[i+1][j][k].isdone)//8
						||(!vp_[i][j][k].isdone && !vp_[i][j][k+1].isdone && !vp_[i+1][j+1][k].isdone)//9
						||(!vp_[i+1][j+1][k].isdone && !vp_[i+1][j+1][k+1].isdone && !vp_[i][j][k+1].isdone)//10
						||(!vp_[i+1][j][k].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i][j+1][k].isdone)//11
						||(!vp_[i][j+1][k].isdone && !vp_[i][j+1][k+1].isdone && !vp_[i+1][j][k].isdone))
					{
						if(!vp_[i][j][k].isdone && !vp_[i+1][j][k].isdone && !vp_[i][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
							tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k+1;
							tp[4][0]=i+1;tp[4][1]=j+1;tp[4][2]=k+1;
						}//1
						else if(!vp_[i][j+1][k].isdone && !vp_[i+1][j+1][k].isdone && !vp_[i+1][j][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
							tp[4][0]=i;tp[4][1]=j;tp[4][2]=k+1;
						}//2
						else if(!vp_[i][j+1][k+1].isdone && !vp_[i+1][j+1][k+1].isdone && !vp_[i][j][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k;
							tp[4][0]=i+1;tp[4][1]=j;tp[4][2]=k;
						}//3
						else if(!vp_[i][j][k+1].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i][j+1][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k+1;
							tp[4][0]=i+1;tp[4][1]=j+1;tp[4][2]=k;
						}//4
						else if(!vp_[i][j][k].isdone && !vp_[i][j+1][k].isdone && !vp_[i+1][j][k+1].isdone)
						{
							//tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							//tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							//tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
							//tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;
							tp[4][0]=i+1;tp[4][1]=j+1;tp[4][2]=k+1;
						}//5
						else if(!vp_[i+1][j][k+1].isdone && !vp_[i+1][j+1][k+1].isdone && !vp_[i][j][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
							tp[4][0]=i;tp[4][1]=j+1;tp[4][2]=k;
						}//6
						else if(!vp_[i+1][j][k].isdone && !vp_[i+1][j+1][k].isdone && !vp_[i][j][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k+1;
							tp[4][0]=i;tp[4][1]=j+1;tp[4][2]=k+1;
						}//7
						else if(!vp_[i][j][k+1].isdone && !vp_[i][j+1][k+1].isdone && !vp_[i+1][j][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k;
							tp[4][0]=i+1;tp[4][1]=j+1;tp[4][2]=k;
						}//8
						else if(!vp_[i][j][k].isdone && !vp_[i][j][k+1].isdone && !vp_[i+1][j+1][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
							tp[4][0]=i+1;tp[4][1]=j+1;tp[4][2]=k+1;
						}//9
						else if(!vp_[i+1][j+1][k].isdone && !vp_[i+1][j+1][k+1].isdone && !vp_[i][j][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k;
							tp[4][0]=i;tp[4][1]=j;tp[4][2]=k;
						}//10
						else if(!vp_[i+1][j][k].isdone && !vp_[i+1][j][k+1].isdone && !vp_[i][j+1][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
							tp[3][0]=i;tp[3][1]=j;tp[3][2]=k+1;
							tp[4][0]=i;tp[4][1]=j+1;tp[4][2]=k+1;
						}//11
						else if(!vp_[i][j+1][k].isdone && !vp_[i][j+1][k+1].isdone && !vp_[i+1][j][k].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k+1;
							tp[4][0]=i+1;tp[4][1]=j;tp[4][2]=k+1;
						}//12
						for(ii=0;ii<5;ii++)
						{
							if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
							{
								vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
								verts_[vertnumber].x=tp[ii][0];
								verts_[vertnumber].y=tp[ii][1];
								verts_[vertnumber].z=tp[ii][2];
								vertnumber++;
							}
						}
						faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
						faces_[facenumber].b=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
						faces_[facenumber++].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
						faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
						faces_[facenumber].b=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
						faces_[facenumber++].c=vertseq[tp[3][0]][tp[3][1]][tp[3][2]];
						faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
						faces_[facenumber].b=vertseq[tp[4][0]][tp[4][1]][tp[4][2]];
						faces_[facenumber++].c=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
						
					}//no.6 12-2

				}//total5
				
				else if(sumtype==6)
				{
					if((!vp_[i][j][k].isdone && !vp_[i+1][j][k].isdone)
						||(!vp_[i][j+1][k].isdone && !vp_[i+1][j+1][k].isdone)
						||(!vp_[i][j+1][k+1].isdone && !vp_[i+1][j+1][k+1].isdone)
						||(!vp_[i][j][k+1].isdone && !vp_[i+1][j][k+1].isdone)
						||(!vp_[i][j][k].isdone && !vp_[i][j+1][k].isdone)
						||(!vp_[i+1][j][k+1].isdone && !vp_[i+1][j+1][k+1].isdone)
						||(!vp_[i+1][j][k].isdone && !vp_[i+1][j+1][k].isdone)
						||(!vp_[i][j][k+1].isdone && !vp_[i][j+1][k+1].isdone)
						||(!vp_[i][j][k].isdone && !vp_[i][j][k+1].isdone)
						||(!vp_[i+1][j+1][k].isdone && !vp_[i+1][j+1][k+1].isdone)
						||(!vp_[i+1][j][k].isdone && !vp_[i+1][j][k+1].isdone)
						||(!vp_[i][j+1][k].isdone && !vp_[i][j+1][k+1].isdone))
					{
						if(!vp_[i][j][k].isdone && !vp_[i+1][j][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
						}//1
						else if(!vp_[i][j+1][k].isdone && !vp_[i+1][j+1][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;
						}//2
						else if(!vp_[i][j+1][k+1].isdone && !vp_[i+1][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k;
						}//3
						else if(!vp_[i][j][k+1].isdone && !vp_[i+1][j][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
						}//4
						else if(!vp_[i][j][k].isdone && !vp_[i][j+1][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
							tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k;
						}//5
						else if(!vp_[i+1][j][k+1].isdone && !vp_[i+1][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
						}//6
						else if(!vp_[i+1][j][k].isdone && !vp_[i+1][j+1][k].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k;
						}//7
						else if(!vp_[i][j][k+1].isdone && !vp_[i][j+1][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k+1;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
						}//8
						else if(!vp_[i][j][k].isdone && !vp_[i][j][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
							tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k;
						}//9
						else if(!vp_[i+1][j+1][k].isdone && !vp_[i+1][j+1][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
						}//10
						else if(!vp_[i+1][j][k].isdone && !vp_[i+1][j][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k;
						}//11
						else if(!vp_[i][j+1][k].isdone && !vp_[i][j+1][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[3][0]=i;tp[3][1]=j;tp[3][2]=k+1;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
						}//12
						for(ii=0;ii<4;ii++)
						{
							if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
							{
								vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
								verts_[vertnumber].x=tp[ii][0];
								verts_[vertnumber].y=tp[ii][1];
								verts_[vertnumber].z=tp[ii][2];
								vertnumber++;
							}
						}
						faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
						faces_[facenumber].b=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
						faces_[facenumber++].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
						faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
						faces_[facenumber].b=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
						faces_[facenumber++].c=vertseq[tp[3][0]][tp[3][1]][tp[3][2]];
						
					}//no.2 12	
					
					else if((!vp_[i][j][k].isdone && !vp_[i+1][j+1][k+1].isdone)
						||(!vp_[i+1][j][k].isdone && !vp_[i][j+1][k+1].isdone)
						||(!vp_[i][j+1][k].isdone && !vp_[i+1][j][k+1].isdone)
						||(!vp_[i+1][j+1][k].isdone && !vp_[i][j][k+1].isdone))
					{
						if(!vp_[i][j][k].isdone && !vp_[i+1][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
							tp[3][0]=i;tp[3][1]=j;tp[3][2]=k+1;
							tp[4][0]=i;tp[4][1]=j+1;tp[4][2]=k;
							tp[5][0]=i+1;tp[5][1]=j;tp[5][2]=k;
						}//1
						else if(!vp_[i+1][j][k].isdone && !vp_[i][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
							tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k+1;
							tp[4][0]=i;tp[4][1]=j;tp[4][2]=k;
							tp[5][0]=i+1;tp[5][1]=j+1;tp[5][2]=k;
						}//2
						else if(!vp_[i][j+1][k].isdone && !vp_[i+1][j][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
							tp[4][0]=i+1;tp[4][1]=j+1;tp[4][2]=k;
							tp[5][0]=i;tp[5][1]=j;tp[5][2]=k;
						}//3
						else if(!vp_[i+1][j+1][k].isdone && !vp_[i][j][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k+1;
							tp[4][0]=i+1;tp[4][1]=j;tp[4][2]=k;
							tp[5][0]=i;tp[5][1]=j+1;tp[5][2]=k;
						}//4
						for(ii=0;ii<6;ii++)
						{
							if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
							{
								vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
								verts_[vertnumber].x=tp[ii][0];
								verts_[vertnumber].y=tp[ii][1];
								verts_[vertnumber].z=tp[ii][2];
								vertnumber++;
							}
						}
						faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
						faces_[facenumber].b=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
						faces_[facenumber++].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
						faces_[facenumber].a=vertseq[tp[3][0]][tp[3][1]][tp[3][2]];
						faces_[facenumber].b=vertseq[tp[4][0]][tp[4][1]][tp[4][2]];
						faces_[facenumber++].c=vertseq[tp[5][0]][tp[5][1]][tp[5][2]];
					}//no.4 4
					
					else if((!vp_[i][j][k].isdone && !vp_[i+1][j][k+1].isdone)
						||(!vp_[i+1][j][k].isdone && !vp_[i][j][k+1].isdone)
						||(!vp_[i+1][j][k].isdone && !vp_[i+1][j+1][k+1].isdone)
						||(!vp_[i+1][j+1][k].isdone && !vp_[i+1][j][k+1].isdone)
						||(!vp_[i+1][j+1][k].isdone && !vp_[i][j+1][k+1].isdone)
						||(!vp_[i][j+1][k].isdone && !vp_[i+1][j+1][k+1].isdone)
						||(!vp_[i][j+1][k].isdone && !vp_[i][j][k+1].isdone)
						||(!vp_[i][j][k].isdone && !vp_[i][j+1][k+1].isdone)
						||(!vp_[i][j][k+1].isdone && !vp_[i+1][j+1][k+1].isdone)
						||(!vp_[i+1][j][k+1].isdone && !vp_[i][j+1][k+1].isdone)
						||(!vp_[i][j][k].isdone && !vp_[i+1][j+1][k].isdone)
						||(!vp_[i+1][j][k].isdone && !vp_[i][j+1][k].isdone))
					{
						if(!vp_[i][j][k].isdone && !vp_[i+1][j][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k+1;
						}//1
						else if(!vp_[i+1][j][k].isdone && !vp_[i][j][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k;
						}//2
						else if(!vp_[i+1][j][k].isdone && !vp_[i+1][j+1][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k+1;
						}//3
						else if(!vp_[i+1][j+1][k].isdone && !vp_[i+1][j][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k;
						}//4
						else if(!vp_[i+1][j+1][k].isdone && !vp_[i][j+1][k+1].isdone)
						{
							tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[3][0]=i;tp[3][1]=j;tp[3][2]=k+1;
						}//5
						else if(!vp_[i][j+1][k].isdone && !vp_[i+1][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;
							tp[3][0]=i;tp[3][1]=j;tp[3][2]=k;
						}//6
						else if(!vp_[i][j+1][k].isdone && !vp_[i][j][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;
							tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k+1;
						}//7
						else if(!vp_[i][j][k].isdone && !vp_[i][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k;
						}//8
						else if(!vp_[i][j][k+1].isdone && !vp_[i+1][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
							tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
							tp[3][0]=i+1;tp[3][1]=j+1;tp[3][2]=k;
						}//9
						else if(!vp_[i+1][j][k+1].isdone && !vp_[i][j+1][k+1].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
							tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
							tp[3][0]=i;tp[3][1]=j+1;tp[3][2]=k;
						}//10
						else if(!vp_[i][j][k].isdone && !vp_[i+1][j+1][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
							tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
							tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[3][0]=i;tp[3][1]=j;tp[3][2]=k+1;
						}//11
						else if(!vp_[i+1][j][k].isdone && !vp_[i][j+1][k].isdone)
						{
							tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
							tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
							tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;
							tp[3][0]=i+1;tp[3][1]=j;tp[3][2]=k+1;
						}//12
						for(ii=0;ii<4;ii++)
						{
							if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
							{
								vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
								verts_[vertnumber].x=tp[ii][0];
								verts_[vertnumber].y=tp[ii][1];
								verts_[vertnumber].z=tp[ii][2];
								vertnumber++;
							}
						}
						faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
						faces_[facenumber].b=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
						faces_[facenumber++].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
						faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
						faces_[facenumber].b=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
						faces_[facenumber++].c=vertseq[tp[3][0]][tp[3][1]][tp[3][2]];
					}//no.3 12
					
				}//total6
				
				else if(sumtype==7)
				{
					if(!vp_[i][j][k].isdone)
					{
						tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k;
						tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k;
						tp[2][0]=i;tp[2][1]=j;tp[2][2]=k+1;
					}//1
					else if(!vp_[i+1][j][k].isdone)
					{
						tp[0][0]=i;tp[0][1]=j;tp[0][2]=k;
						tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k;		
						tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k+1;
					}//2
					else if(!vp_[i+1][j+1][k].isdone)
					{
						tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k;
						tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k;		
						tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k+1;
					}//3
					else if(!vp_[i][j+1][k].isdone)
					{				
						tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k;
						tp[1][0]=i;tp[1][1]=j;tp[1][2]=k;
						tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k+1;
					}//4
					else if(!vp_[i][j][k+1].isdone)
					{
						tp[0][0]=i+1;tp[0][1]=j;tp[0][2]=k+1;
						tp[1][0]=i;tp[1][1]=j+1;tp[1][2]=k+1;		
						tp[2][0]=i;tp[2][1]=j;tp[2][2]=k;
					}//5
					else if(!vp_[i+1][j][k+1].isdone)
					{
						tp[0][0]=i+1;tp[0][1]=j+1;tp[0][2]=k+1;
						tp[1][0]=i;tp[1][1]=j;tp[1][2]=k+1;		
						tp[2][0]=i+1;tp[2][1]=j;tp[2][2]=k;
					}//6
					else if(!vp_[i+1][j+1][k+1].isdone)
					{
						tp[0][0]=i;tp[0][1]=j+1;tp[0][2]=k+1;
						tp[1][0]=i+1;tp[1][1]=j;tp[1][2]=k+1;			
						tp[2][0]=i+1;tp[2][1]=j+1;tp[2][2]=k;
					}//7
					else if(!vp_[i][j+1][k+1].isdone)
					{
						tp[0][0]=i;tp[0][1]=j;tp[0][2]=k+1;
						tp[1][0]=i+1;tp[1][1]=j+1;tp[1][2]=k+1;		
						tp[2][0]=i;tp[2][1]=j+1;tp[2][2]=k;
					}//8
					for(ii=0;ii<3;ii++)
					{
						if(vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]==-1)
						{
							vertseq[tp[ii][0]][tp[ii][1]][tp[ii][2]]=vertnumber;
							verts_[vertnumber].x=tp[ii][0];
							verts_[vertnumber].y=tp[ii][1];
							verts_[vertnumber].z=tp[ii][2];
							vertnumber++;
						}
					}
					faces_[facenumber].a=vertseq[tp[0][0]][tp[0][1]][tp[0][2]];
					faces_[facenumber].b=vertseq[tp[1][0]][tp[1][1]][tp[1][2]];
					faces_[facenumber++].c=vertseq[tp[2][0]][tp[2][1]][tp[2][2]];
				}//total7
						
			}//every ijk
		}//j
	}//i
	// verts=(vertinfo *)doubleloc(verts,vertnumber*sizeof(vertinfo));
	// faces=(faceinfo *)doubleloc(faces,facenumber*sizeof(faceinfo));
	verts_.resize(vertnumber);
	faces_.resize(facenumber);
	for(i=0;i<vertnumber;i++)
	{
		// verts_[i].atomid=vp_[int(verts_[i].x)][int(verts_[i].y)][int(verts_[i].z)].atomid;
		verts_[i].iscont=false;
		if(vp_[int(verts_[i].x)][int(verts_[i].y)][int(verts_[i].z)].isbound)
			verts_[i].iscont=true;
	}
	// for(i=0;i<plength;i++)
	// {
	// 	for(j=0;j<pwidth;j++)
	// 	{
	// 		delete[]vertseq[i][j];
	// 	}
	// }
	
	// for(i=0;i<plength;i++)
	// {
	// 	delete[]vertseq[i];
	// }
	// delete[]vertseq;
}



