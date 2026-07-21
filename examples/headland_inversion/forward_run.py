from thetis import *
from firedrake import *
from firedrake import VTKFile
import geopandas as gpd
from model_config import construct_solver
from shapely.geometry import Point
from mpi4py import MPI

# ---------------------------------------- Step 1: set up mesh and ground truth ----------------------------------------
# Create an ensemble communicator. M sets the number of MPI ranks used for each ensemble member
M = 2
ensemble = Ensemble(MPI.COMM_WORLD, M)
ensemble_rank = ensemble.ensemble_rank

# Create an output directory for each ensemble rank
pwd = os.path.abspath(os.path.dirname(__file__))
output_dir_forward = os.path.join(pwd, 'outputs', 'outputs_forward', f'member_{ensemble_rank}')

# In this case each member will start its run at a slightly different point along the sinusoidal forcing.
# ach member then stores observations at its assigned station. This is meant to approximate a scenario where multiple
# velocity/elevation measurement surveys may take place in the same region at different times.
time_offset = ensemble_rank * 800.0

# Passing some new parameters to construct_solver (comm, distribution_parameters, time_offset, station_index)
solver_obj, update_forcings = construct_solver(
    output_directory=output_dir_forward,
    store_station_time_series=True,
    no_exports=False,
    comm=ensemble.comm,                                         # used to sync up operations across ensemble members
    distribution_parameters={'partitioner_type': 'simple'},     # for splitting mesh consistently across threads
    time_offset=time_offset,
    station_index=ensemble_rank,                                # each ensemble member will use different station index
)

mesh2d = solver_obj.mesh2d
options = solver_obj.options
manning_2d = solver_obj.fields.manning_2d
elev_init_2d = solver_obj.fields.elev_2d

coordinates = mesh2d.coordinates.dat.data[:]
x, y = coordinates[:, 0], coordinates[:, 1]
lx = mesh2d.comm.allreduce(numpy.max(x), MPI.MAX)
ly = mesh2d.comm.allreduce(numpy.max(y), MPI.MAX)

# Create a FunctionSpace on the mesh (corresponds to Manning)
V = get_functionspace(mesh2d, 'CG', 1)

# Load the shapefile
shapefile_path = os.path.join(pwd, 'inputs', 'bed_classes.shp')
gdf = gpd.read_file(shapefile_path)
polygons_by_id = gdf.groupby('id')

sediment_to_manning = {
    'ROCK': 0.0420,
    'SAND': 0.0171,
    'SANDY CLAY': 0.0132,
    'MUDDY SAND': 0.0163,
    'CLAY': 0.0100
}

mask_values = []
masks = [Function(V) for _ in range(len(polygons_by_id))]
m_true = []

for i, (region_id, group) in enumerate(polygons_by_id):
    multi_polygon = group.union_all()

    # Get the sediment type for this region (assuming one sediment type per ID)
    sediment_type = group['Sediment'].iloc[0]
    manning_value = sediment_to_manning.get(sediment_type, None)
    values = []

    for (x_, y_) in zip(x, y):
        # Check if the point is inside the multi-polygon
        point = Point(x_, y_)
        if multi_polygon.buffer(1).contains(point):
            values.append(1)
        else:
            values.append(0)

    mask_values.append(values)
    m_true.append(domain_constant(manning_value, mesh2d))

overlap_counts = numpy.zeros(len(x))

for values in mask_values:
    overlap_counts += numpy.array(values)

for values in mask_values:
    for i in range(len(values)):
        if overlap_counts[i] > 1:
            values[i] /= overlap_counts[i]

for mask, values in zip(masks, mask_values):
    mask.dat.data[:] = values

manning_2d.assign(0)
for m_, mask_ in zip(m_true, masks):
    manning_2d += m_ * mask_

# Overwrite the default initial manning value
# need to pass ensemble.comm for export to work properly
VTKFile(os.path.join(output_dir_forward, 'manning_init.pvd'), comm=ensemble.comm).write(manning_2d)

# print statements to check setup
if ensemble.comm.rank == 0:
    print(f'Ensemble member {ensemble_rank + 1}/{ensemble.ensemble_size}: '
          f'time offset = {time_offset}; detector station index = {ensemble_rank}')
print_output('Exporting to ' + solver_obj.options.output_directory)

print_output('Solving the forward problem...')
solver_obj.iterate(update_forcings=update_forcings)
