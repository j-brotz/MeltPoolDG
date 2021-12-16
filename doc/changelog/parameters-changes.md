# Parameters changelog
All notable changes of the input parameters will be documented in this file.

## 2021-12-10
- Add new parameter to distinguish how interfacial forces are calculated: 
```json
{
  "recoil pressure" : {
     "interface distributed flux type": "continuous|interface value"
  }
}
```

## 2021-12-16
- Add new option for Dirac delta approximation type
```json
{
  "dirac delta function approximation": {
    "type": "delta_weighted_consistent_with_evaporation"
  }
}
```