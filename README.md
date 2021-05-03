# COMF

**COMF** is a web application that helps the end user to interact, and model isothermal kinetic data obtained by isothermal thermogravimetric analyzers (TGA).



**COMF** is a Python-based web application to obtain kinetic information from thermogravimetric analysis (TGA) data to characterize solid materials, particularly of pharmaceutical, petrochemical, food and environmental interest. It basically provides a full kinetic analysis of these data, including the determination of the **reaction model** and calculation of the **activation enthalpy** for isothermal solid-state reactions. It can be used to investigate physicochemical phenomena, such as phase transitions, desolvation/dehydration of pharmaceutical solvates/hydrates, absorption, adsorption and desorption, thermal decomposition, and solid-gas reactions, e.g. oxidation or reduction.

The application is based on the work of [Y. Vasilopoulos et al.](https://www.mdpi.com/2073-4352/10/2/139/htm). The algorithm of the publication has been corrected without loss of generality and the relative error of residuals is used as the main metrics.

## Installation

Clone this repository using [GitHub](https://help.github.com/en/enterprise/2.13/user/articles/cloning-a-repository)

```bash
git clone https://github.com/vasilogi/COMF.git
```



This app has been developed in **Python 3.8.8**.



In the main directory run

```bash
pip install -r requirements.txt
```

## Usage

git clone https://github.com/vasilogi/COMF.git

### CSV data format

Add your data to the [data](./data) directory in a CSV format. The CSV file must include at least 6 columns, namely **mass**, **mass units**, **time**, **time units**, **temperature**, **temperature units**. **conversion** is optional. Filenames do not matter but headers do. To properly prepare your csv data file check out the existing [example](./data/80_0_Celcius.csv) and below:

| mass        | mass units | conversion  | time        | time units | temperature | temperature units |
| ----------- | :--------- | ----------- | ----------- | ---------- | ----------- | ----------------- |
| 9.86114426  | mg         | 0.077142078 | 17.27410444 | min        | 320         | Kelvin            |
| 9.856718586 | mg         | 0.079600786 | 18.43308332 | min        | 320         | Kelvin            |
| 9.835847846 | mg         | 0.091195641 | 19.59206219 | min        | 320         | Kelvin            |
| 9.842446253 | mg         | 0.08752986  | 20.75104107 | min        | 320         | Kelvin            |

### Run 

```bas
python main.py
```

The app will run on your local server http://127.0.0.1:8050/

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
