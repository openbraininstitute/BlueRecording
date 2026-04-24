# Objective CSD calculation

BlueRecording permits the calculation of the "Objective CSD" metrics defined in the forthcoming paper "iCSD produces spurious results in dense electrode arrays". Two variants of the metric are defined in this paper: Objective Sphere CSD ($o_sCSD$) and Objective Disk CSD ($o_DCSD$). In addition, BlueRecording supports a third variant, "Objective Plane CSD", which is effectively $o_DCSD$ with a disk of infinite radius.

## Objective CSD electrodes in the input CSV file
In the CSV file used as an input to `bluerecording.weights.initialize_h5_file()`, electrodes using the objective CSD methods can be specified by setting the value of the `type` column to `ObjectiveCSD_Sphere`, `ObjectiveCSD_Disk`, or `ObjectiveCSD_Plane`.

Geometry parameters are specified via optional `radius` and `thickness` columns in the CSV. Leave cells empty (or omit the columns entirely) to use the defaults:

- **ObjectiveCSD_Sphere**: `radius` sets the sphere radius in µm (default: 10 µm).
- **ObjectiveCSD_Disk**: `radius` sets the disk radius in µm (default: 500 µm); `thickness` sets the disk half-height in µm (default: estimated from inter-electrode spacing).
- **ObjectiveCSD_Plane**: `thickness` sets the plane half-height in µm (default: estimated from inter-electrode spacing).

*Note that incorrect results may be produced for ObjectiveCSD_Disk and ObjectiveCSD_Plane if the electrodes are not equally spaced and in a single line.*

### Example CSV

```csv
name,x,y,z,layer,region,type,radius,thickness
0,100,200,0,L5,S1FL,ObjectiveCSD_Sphere,15,
1,100,200,50,L5,S1FL,ObjectiveCSD_Disk,500,25
2,100,200,100,L5,S1FL,ObjectiveCSD_Plane,,30
```

## Calculating disk thickness from a subset of electrodes
It is possible that your electrodes csv file (and consequently your h5 weights file) will contain multiple electrode arrays from which you wish to calculate the objective disk or objective plane CSD. In this case, you can supply the argument `objective_csd_array_indices` to the function `bluerecording.weights.write_h5_file()`. `objective_csd_array_indices` is a list of strings, with each string having the form `"a:b"`, where a and b are the start and end indices in the csv file of one of the electrode arrays. If the thickness of the disks/planes is not specified in the CSV, BlueRecording will estimate the thickness for each array separately based on the `objective_csd_array_indices` argument.
