# Benchmarking the Python and MATLAB implementation
Below we will compare the Python implementation with the MATLAB one in terms of the metric values and the runtime of MoDE for different datasets.

### Distance, Correlation, and Order metrics
The table below shows the metrics accuracy results for Python and MATLAB implementation of MoDE.

<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th colspan="3">Python</th>
    <th colspan="3">MATLAB</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Metrics</td>
    <td>R_d</td>
    <td>R_o</td>
    <td>R_c</td>
    <td>R_d</td>
    <td>R_o</td>
    <td>R_c</td>
  </tr>
  <tr>
    <td>Small Stock</td>
    <td>0.708</td>
    <td>0.955</td>
    <td>0.864</td>
    <td>0.708</td>
    <td>0.960</td>
    <td>0.867</td>
  </tr>
  <tr>
    <td>Big Stock</td>
    <td>0.707</td>
    <td>0.952</td>
    <td>0.89</td>
    <td>0.707</td>
    <td>0.953</td>
    <td>0.894</td>
  </tr>
</tbody>
</table>

### Runtime comparison
The table below shows the runtime comparison (in seconds) for different datasets. The experiments were done on a 2.5 GHz 14-Core Intel Xenon with 256 GB of RAM.

<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th># points</th>
    <th>Python</th>
    <th>MATLAB</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Small Stock<br></td>
    <td>436</td>
    <td>8.14</td>
    <td>3.579</td>
  </tr>
  <tr>
    <td>Big Stock</td>
    <td>2256</td>
    <td>133.5</td>
    <td>69.768</td>
  </tr>
  <tr>
    <td>Breast Cancer</td>
    <td>569</td>
    <td>11.1</td>
    <td>4.56</td>
  </tr>
  <tr>
    <td>cifar-10 (subset)</td>
    <td>8000</td>
    <td>838.81</td>
    <td>409.52</td>
  </tr>
  <tr>
    <td>EEG</td>
    <td>11853</td>
    <td>2594.15</td>
    <td>583.323</td>
  </tr>
  <tr>
    <td>heart beat</td>
    <td>14545</td>
    <td>1529.77</td>
    <td>837.88</td>
  </tr>
  <tr>
    <td>madelon</td>
    <td>2080</td>
    <td>147.25</td>
    <td>65.75</td>
  </tr>
  <tr>
    <td>MNIST (subset)</td>
    <td>2000</td>
    <td>96.02</td>
    <td>23.89</td>
  </tr>
</tbody>
</table>

## Speed-Up Benchmarks

### Swiss Roll Dataset with 10K points
<table>
<thead>
  <tr>
    <th>Method</th>
    <th> Runtime( in seconds) </th>
    <th>R_d</th>
    <th>R_c</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>LLE<br></td>
    <td>12</td>
    <td>0.0018824</td>
    <td>0.99747</td>
  </tr>
  <tr>
    <td>LTSA</td>
    <td>33</td>
    <td>0.0022112</td>
    <td>0.99695</td>
  </tr>
  <tr>
    <td>Hessain LLE</td>
    <td>15</td>
    <td>0.0022112</td>
    <td>0.99695</td>
  </tr>
  <tr>
    <td>Isomap</td>
    <td>71</td>
    <td>0.90944</td>
    <td>0.9981</td>
  </tr>
  <tr>
    <td>Spectral Embedding</td>
    <td>14</td>
    <td>0.00026815</td>
    <td>0.99917</td>
  </tr>
  <tr>
    <td>t - SNE</td>
    <td>83</td>
    <td>0.65674</td>
    <td>0.99588</td>
  </tr>
  <tr>
    <td>MoDE</td>
    <td>36</td>
    <td>0.78958</td>
    <td>0.99933</td>
  </tr>
</tbody>
</table>

