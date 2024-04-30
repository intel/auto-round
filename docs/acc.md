## Comparison with other methods
To ensure a fair comparison as much as possible and alleviate overfitting in perplexity evaluation on Wikitext or C4, we utilized 512 samples from NeelNanda/pile-10k for all methods during calibration unless explicitly stated. For wikitext2/ptb-new/c4-new ppl, we follow the code of gptq and set the sequence length to 2048. For lm-eval wikitext ppl, we adopt lm-eval. The lm-eval-harness git id we used in the following is 008fc2a23245c40384f2312718433eeb1e0f87a9 and we evaluated on qdq fake models.

Due to memory constraints, we maintained the original sequence length of 512 for AWQ, while for GPTQ，Omniquant and our approach, a sequence length of 2048 is used. And HQQ is a data free method, no need to calibrate.

For GPTQ, we have enabled act-order and true-seqential, and also activated static group in scenarios where group_size!=-1. The notation GPTQ* indicates that we adjusted the random seed or data preprocessing to address issues related to the non-positive definite Hessian matrix or other issues.

For Omniquant, we adhere to the official settings, which include running for 20 epochs and disabling 'let'. We conducted calibration tests using sample sizes of 512 and 128, as well as a sample size of 512 with a batch size of 4. Our findings show that using a sample size of 512 typically results in comparable or slight higher performance for models <=13B. Therefore, we present the results based on the sample size of 512. For 70B models, due the the NAN loss issue and to reduce the tuning cost, we adopted 128 samples for calibration.

For AutoRound, we used the default setting, iters 200, enable_quanted_input and enable_minmax_tuning, both the lr and minmax_lr are set to 1/iters,i.e. 5e-3.

With these configurations, the tuning costs for GPTQ, AWQ, and ours are similar, while HQQ is much faster and Omniquant is noticebal slower.

</br>

### 1. Accuracies $\uparrow$ across 11 tasks(0-shot) of LLaMA and Mistral models at W4G-1.
<table border="1">
    <tr>
        <td></td>
        <td></td>
        <td>Mmlu</td>
        <td>Lamb.</td>
        <td>Hella.</td>
        <td>Wino.</td>
        <td>Piqa</td>
        <td>Truth.</td>
        <td>Open.</td>
        <td>Boolq</td>
        <td>RTE</td>
        <td>ARC-e</td>
        <td>ARC-c.</td>
        <td>Avg.</td>
    </tr>
    <tr>
        <td rowspan="7">Mistral-7B</td>
        <td>FP16</td>
        <td>61.35</td>
        <td>75.68</td>
        <td>61.27</td>
        <td>74.03</td>
        <td>80.79</td>
        <td>28.03</td>
        <td>32.80</td>
        <td>83.67</td>
        <td>67.51</td>
        <td>80.81</td>
        <td>50.34</td>
        <td>63.30</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>55.92</td>
        <td>66.10</td>
        <td>59.01</td>
        <td>71.35</td>
        <td>80.14</td>
        <td>24.85</td>
        <td>29.00</td>
        <td>79.17</td>
        <td>57.76</td>
        <td>77.95</td>
        <td>45.99</td>
        <td>58.84</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>58.22</td>
        <td>73.45</td>
        <td>59.47</td>
        <td>74.03</td>
        <td>80.20</td>
        <td>26.93</td>
        <td>31.00</td>
        <td>81.50</td>
        <td>64.98</td>
        <td>78.24</td>
        <td>47.01</td>
        <td>61.37</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>57.20</td>
        <td>71.45</td>
        <td>59.21</td>
        <td>73.64</td>
        <td>79.43</td>
        <td>25.34</td>
        <td>30.40</td>
        <td>82.69</td>
        <td>68.95</td>
        <td>79.25</td>
        <td>47.44</td>
        <td>61.36</td>
    </tr>
    <tr>
        <td>HQQ</td>
        <td>52.65 </td>
        <td>66.58 </td>
        <td>59.09 </td>
        <td>70.56 </td>
        <td>79.60 </td>
        <td>23.13 </td>
        <td>27.80 </td>
        <td>80.03 </td>
        <td>59.57 </td>
        <td>77.02 </td>
        <td>46.33 </td>
        <td>58.40 </td>
    </tr>
    <tr>
        <td>Omniquant</td>
        <td>57.52 </td>
        <td>70.00 </td>
        <td>60.27 </td>
        <td>72.93 </td>
        <td>79.87 </td>
        <td>23.99 </td>
        <td>30.80 </td>
        <td>81.53 </td>
        <td>63.90 </td>
        <td>78.54 </td>
        <td>46.42 </td>
        <td>60.52 </td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>59.52</td>
        <td>73.76</td>
        <td>60.75</td>
        <td>73.32</td>
        <td>80.09</td>
        <td>27.17</td>
        <td>33.00</td>
        <td>82.02</td>
        <td>66.07</td>
        <td>80.47</td>
        <td>49.49</td>
        <td><b>62.33</td>
    </tr>
    <tr>
        <td rowspan="7">V2-7B</td>
        <td>FP16</td>
        <td>42.69</td>
        <td>73.90</td>
        <td>57.15</td>
        <td>68.90</td>
        <td>78.07</td>
        <td>25.21</td>
        <td>31.40</td>
        <td>77.74</td>
        <td>62.82</td>
        <td>76.35</td>
        <td>43.52</td>
        <td>57.98</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>36.87</td>
        <td>67.96</td>
        <td>55.63</td>
        <td>68.51</td>
        <td>76.82</td>
        <td>26.19</td>
        <td>30.60</td>
        <td>73.64</td>
        <td>58.84</td>
        <td>74.07</td>
        <td>41.30</td>
        <td>55.49</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>39.66</td>
        <td>71.92</td>
        <td>55.89</td>
        <td>68.03</td>
        <td>77.58</td>
        <td>25.09</td>
        <td>30.20</td>
        <td>76.67</td>
        <td>62.09</td>
        <td>75.55</td>
        <td>41.72</td>
        <td>56.76</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>40.24</td>
        <td>71.20</td>
        <td>56.26</td>
        <td>69.61</td>
        <td>76.93</td>
        <td>26.07</td>
        <td>32.60</td>
        <td>77.31</td>
        <td>63.18</td>
        <td>75.00</td>
        <td>41.30</td>
        <td>57.25</td>
    </tr>
    <tr>
        <td>HQQ</td>
        <td>28.94 </td>
        <td>43.96 </td>
        <td>48.43 </td>
        <td>59.43 </td>
        <td>71.82 </td>
        <td>23.62 </td>
        <td>24.80 </td>
        <td>52.11 </td>
        <td>53.79 </td>
        <td>64.90 </td>
        <td>34.73 </td>
        <td>46.05 </td>
    </tr>
    <tr>
        <td>Omniquant</td>
        <td>39.82 </td>
        <td>71.45 </td>
        <td>55.76 </td>
        <td>67.56 </td>
        <td>76.88 </td>
        <td>25.09 </td>
        <td>30.80 </td>
        <td>76.15 </td>
        <td>64.98 </td>
        <td>74.12 </td>
        <td>40.19 </td>
        <td>56.62 </td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>39.97</td>
        <td>71.63</td>
        <td>56.52</td>
        <td>68.43</td>
        <td>77.91</td>
        <td>25.70</td>
        <td>31.60</td>
        <td>76.18</td>
        <td>65.70</td>
        <td>76.01</td>
        <td>42.58</td>
        <td><b>57.48</td>
    </tr>
    <tr>
        <td rowspan="7">V2-13B</td>
        <td>FP16</td>
        <td>52.86</td>
        <td>76.77</td>
        <td>60.04</td>
        <td>72.14</td>
        <td>79.05</td>
        <td>25.95</td>
        <td>35.20</td>
        <td>80.55</td>
        <td>65.34</td>
        <td>79.38</td>
        <td>48.38</td>
        <td>61.42</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>50.37</td>
        <td>74.35</td>
        <td>59.12</td>
        <td>71.98</td>
        <td>79.00</td>
        <td>24.85</td>
        <td>33.00</td>
        <td>81.77</td>
        <td>64.98</td>
        <td>79.08</td>
        <td>46.59</td>
        <td>60.46</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>51.14</td>
        <td>75.37</td>
        <td>59.14</td>
        <td>72.06</td>
        <td>78.02</td>
        <td>25.34</td>
        <td>32.20</td>
        <td>80.46</td>
        <td>62.09</td>
        <td>77.36</td>
        <td>44.54</td>
        <td>59.79</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>51.16</td>
        <td>75.98</td>
        <td>59.51</td>
        <td>70.80</td>
        <td>78.40</td>
        <td>25.21</td>
        <td>34.60</td>
        <td>78.26</td>
        <td>66.79</td>
        <td>79.12</td>
        <td>46.59</td>
        <td>60.58</td>
    </tr>
    <tr>
        <td>HQQ</td>
        <td>35.92 </td>
        <td>49.54 </td>
        <td>46.27 </td>
        <td>58.01 </td>
        <td>72.47 </td>
        <td>23.99 </td>
        <td>19.80 </td>
        <td>61.77 </td>
        <td>51.26 </td>
        <td>62.84 </td>
        <td>33.19 </td>
        <td>46.82 </td>
    </tr>
    <tr>
        <td>Omniquant</td>
        <td>51.01 </td>
        <td>75.45 </td>
        <td>59.48 </td>
        <td>71.74 </td>
        <td>78.94 </td>
        <td>24.60 </td>
        <td>33.20 </td>
        <td>77.37 </td>
        <td>66.07 </td>
        <td>78.75 </td>
        <td>46.76 </td>
        <td>60.31 </td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>52.30</td>
        <td>75.96</td>
        <td>59.79</td>
        <td>72.30</td>
        <td>78.84</td>
        <td>25.58</td>
        <td>34.00</td>
        <td>80.15</td>
        <td>66.79</td>
        <td>79.38</td>
        <td>48.12</td>
        <td><b>61.20</td>
    </tr>
    <tr>
        <td rowspan="7">V2-70B</td>
        <td>FP16</td>
        <td>66.23</td>
        <td>79.64</td>
        <td>64.77</td>
        <td>77.98</td>
        <td>82.15</td>
        <td>30.60</td>
        <td>37.20</td>
        <td>83.70</td>
        <td>67.87</td>
        <td>82.70</td>
        <td>54.44</td>
        <td>66.12</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>63.85</td>
        <td>77.62</td>
        <td>63.38</td>
        <td>76.72</td>
        <td>81.50</td>
        <td>28.89</td>
        <td>37.80</td>
        <td>83.39</td>
        <td>68.23</td>
        <td>81.99</td>
        <td>54.10</td>
        <td>65.22</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>64.81</td>
        <td>79.27</td>
        <td>63.86</td>
        <td>76.87</td>
        <td>81.61</td>
        <td>31.46</td>
        <td>36.40</td>
        <td>82.23</td>
        <td>70.04</td>
        <td>82.53</td>
        <td>54.18</td>
        <td>65.75</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>65.08</td>
        <td>78.77</td>
        <td>64.14</td>
        <td>77.11</td>
        <td>81.45</td>
        <td>30.48</td>
        <td>37.20</td>
        <td>83.64</td>
        <td>72.92</td>
        <td>82.49</td>
        <td>55.80</td>
        <td><b>66.28</td>
    </tr>
    <tr>
        <td>HQQ</td>
        <td>56.45 </td>
        <td>66.74 </td>
        <td>53.67 </td>
        <td>73.32 </td>
        <td>76.50 </td>
        <td>25.58 </td>
        <td>33.40 </td>
        <td>67.95 </td>
        <td>61.73 </td>
        <td>72.90 </td>
        <td>43.94 </td>
        <td>57.47 </td>
    </tr>
    <tr>
        <td>Omniquant</td>
        <td>64.40 </td>
        <td>79.20 </td>
        <td>63.91 </td>
        <td>76.95 </td>
        <td>81.94 </td>
        <td>31.70 </td>
        <td>37.60 </td>
        <td>82.35 </td>
        <td>69.31 </td>
        <td>82.24 </td>
        <td>54.18 </td>
        <td>65.80 </td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>65.43</td>
        <td>79.55</td>
        <td>64.47</td>
        <td>78.06</td>
        <td>82.10</td>
        <td>30.60</td>
        <td>36.40</td>
        <td>83.91</td>
        <td>71.12</td>
        <td>82.53</td>
        <td>54.78</td>
        <td>66.27</td>
    </tr>
    <tr>
        <td rowspan="5">V1-7B</td>
        <td>FP16</td>
        <td>32.74</td>
        <td>73.53</td>
        <td>56.94</td>
        <td>70.01</td>
        <td>78.67</td>
        <td>22.03</td>
        <td>34.60</td>
        <td>75.08</td>
        <td>66.43</td>
        <td>75.25</td>
        <td>41.81</td>
        <td>57.01</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>31.34</td>
        <td>70.02</td>
        <td>55.35</td>
        <td>69.77</td>
        <td>77.69</td>
        <td>20.32</td>
        <td>32.60</td>
        <td>73.43</td>
        <td>59.57</td>
        <td>74.45</td>
        <td>41.30</td>
        <td>55.08</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>29.06</td>
        <td>71.08</td>
        <td>55.11</td>
        <td>70.01</td>
        <td>77.37</td>
        <td>20.93</td>
        <td>32.20</td>
        <td>72.69</td>
        <td>63.90</td>
        <td>74.66</td>
        <td>41.64</td>
        <td>55.33</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>33.33</td>
        <td>70.81</td>
        <td>55.98</td>
        <td>68.27</td>
        <td>78.07</td>
        <td>21.18</td>
        <td>31.40</td>
        <td>74.37</td>
        <td>64.62</td>
        <td>74.03</td>
        <td>41.21</td>
        <td>55.75</td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>31.80</td>
        <td>71.96</td>
        <td>56.57</td>
        <td>69.53</td>
        <td>79.00</td>
        <td>21.91</td>
        <td>33.20</td>
        <td>75.72</td>
        <td>66.79</td>
        <td>74.83</td>
        <td>43.09</td>
        <td><b>56.76</td>
    </tr>
    <tr>
        <td rowspan="5">V1-13B</td>
        <td>FP16</td>
        <td>44.21</td>
        <td>76.21</td>
        <td>59.92</td>
        <td>72.77</td>
        <td>79.16</td>
        <td>25.70</td>
        <td>33.20</td>
        <td>77.89</td>
        <td>70.76</td>
        <td>77.40</td>
        <td>46.42</td>
        <td>60.33</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>39.57</td>
        <td>70.93</td>
        <td>58.82</td>
        <td>71.98</td>
        <td>78.02</td>
        <td>24.85</td>
        <td>32.00</td>
        <td>78.20</td>
        <td>66.43</td>
        <td>75.67</td>
        <td>44.62</td>
        <td>58.28</td>
    </tr>
    <tr>
        <td>GPTQ*</td>
        <td>40.01</td>
        <td>74.67</td>
        <td>58.92</td>
        <td>71.03</td>
        <td>78.45</td>
        <td>26.44</td>
        <td>33.60</td>
        <td>77.09</td>
        <td>68.23</td>
        <td>76.85</td>
        <td>44.97</td>
        <td>59.12</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>44.56</td>
        <td>74.13</td>
        <td>59.13</td>
        <td>71.27</td>
        <td>78.94</td>
        <td>25.83</td>
        <td>33.20</td>
        <td>76.42</td>
        <td>66.06</td>
        <td>76.89</td>
        <td>46.67</td>
        <td>59.37</td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>43.94</td>
        <td>75.82</td>
        <td>59.51</td>
        <td>72.22</td>
        <td>78.78</td>
        <td>25.70</td>
        <td>32.80</td>
        <td>77.34</td>
        <td>67.51</td>
        <td>76.47</td>
        <td>46.67</td>
        <td><b>59.71</td>
    </tr>
    <tr>
        <td rowspan="5">V1-30B</td>
        <td>FP16</td>
        <td>55.14</td>
        <td>77.55</td>
        <td>63.33</td>
        <td>75.85</td>
        <td>81.12</td>
        <td>28.27</td>
        <td>36.00</td>
        <td>82.78</td>
        <td>66.79</td>
        <td>80.39</td>
        <td>52.90</td>
        <td>63.65</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>53.05</td>
        <td>75.65</td>
        <td>62.08</td>
        <td>74.82</td>
        <td>80.09</td>
        <td>25.95</td>
        <td>35.80</td>
        <td>81.87</td>
        <td>63.54</td>
        <td>79.76</td>
        <td>50.26</td>
        <td>62.08</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>53.04</td>
        <td>77.22</td>
        <td>61.95</td>
        <td>73.80</td>
        <td>80.69</td>
        <td>27.29</td>
        <td>34.60</td>
        <td>81.07</td>
        <td>66.06</td>
        <td>78.79</td>
        <td>49.15</td>
        <td>62.15</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>54.13</td>
        <td>76.77</td>
        <td>62.78</td>
        <td>74.11</td>
        <td>81.07</td>
        <td>27.78</td>
        <td>35.00</td>
        <td>82.66</td>
        <td>67.15</td>
        <td>79.97</td>
        <td>51.71</td>
        <td>63.01</td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>54.72</td>
        <td>77.84</td>
        <td>62.91</td>
        <td>75.06</td>
        <td>80.69</td>
        <td>26.68</td>
        <td>36.40</td>
        <td>82.60</td>
        <td>66.79</td>
        <td>80.13</td>
        <td>52.13</td>
        <td><b>63.27</td>
    </tr>
    <tr>
        <td rowspan="5">V1-65B</td>
        <td>FP16</td>
        <td>59.79</td>
        <td>79.12</td>
        <td>64.53</td>
        <td>77.35</td>
        <td>81.23</td>
        <td>27.91</td>
        <td>38.00</td>
        <td>84.86</td>
        <td>69.68</td>
        <td>81.36</td>
        <td>52.82</td>
        <td>65.15</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>58.74</td>
        <td>76.42</td>
        <td>64.12</td>
        <td>76.72</td>
        <td>81.01</td>
        <td>29.25</td>
        <td>38.60</td>
        <td>84.13</td>
        <td>70.40</td>
        <td>80.72</td>
        <td>51.88</td>
        <td>64.73</td>
    </tr>
    <tr>
        <td>GPTQ*</td>
        <td>59.10</td>
        <td>78.17</td>
        <td>63.78</td>
        <td>75.69</td>
        <td>81.34</td>
        <td>28.27</td>
        <td>38.40</td>
        <td>83.76</td>
        <td>68.59</td>
        <td>80.98</td>
        <td>51.62</td>
        <td>64.52</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>58.86</td>
        <td>77.37</td>
        <td>63.86</td>
        <td>76.56</td>
        <td>80.85</td>
        <td>28.27</td>
        <td>35.20</td>
        <td>83.94</td>
        <td>71.48</td>
        <td>78.75</td>
        <td>50.94</td>
        <td>64.19</td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>59.21</td>
        <td>79.16</td>
        <td>64.37</td>
        <td>76.64</td>
        <td>81.34</td>
        <td>26.81</td>
        <td>37.80</td>
        <td>84.40</td>
        <td>69.68</td>
        <td>80.98</td>
        <td>51.79</td>
        <td><b>64.74</td>
    </tr>
</table>

</br>

### 2. Accuracies $\uparrow$ across 11 tasks(0-shot) of LLaMA and Mistral models at W4G128.

<table border="1">
    <tr>
        <td></td>
        <td></td>
        <td>Mmlu</td>
        <td>Lamb.</td>
        <td>Hella.</td>
        <td>Wino.</td>
        <td>Piqa</td>
        <td>Truth.</td>
        <td>Open.</td>
        <td>Boolq</td>
        <td>RTE</td>
        <td>ARC-e</td>
        <td>ARC-c.</td>
        <td>Avg.</td>
    </tr>
    <tr>
        <td  rowspan="7">Mistral-7B</td>
        <td>FP16</td>
        <td>61.35</td>
        <td>75.68</td>
        <td>61.27</td>
        <td>74.03</td>
        <td>80.79</td>
        <td>28.03</td>
        <td>32.80</td>
        <td>83.67</td>
        <td>67.51</td>
        <td>80.81</td>
        <td>50.34</td>
        <td>63.30</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>59.72</td>
        <td>74.44</td>
        <td>61.06</td>
        <td>73.40</td>
        <td>80.36</td>
        <td>27.17</td>
        <td>32.60</td>
        <td>83.67</td>
        <td>64.62</td>
        <td>79.63</td>
        <td>49.32</td>
        <td>62.36</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>59.17</td>
        <td>74.52</td>
        <td>60.37</td>
        <td>74.90</td>
        <td>80.58</td>
        <td>26.68</td>
        <td>31.00</td>
        <td>83.33</td>
        <td>67.15</td>
        <td>79.67</td>
        <td>48.12</td>
        <td>62.32</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>60.20</td>
        <td>75.14</td>
        <td>60.43</td>
        <td>73.80</td>
        <td>80.03</td>
        <td>27.05</td>
        <td>30.40</td>
        <td>84.01</td>
        <td>62.09</td>
        <td>80.39</td>
        <td>50.26</td>
        <td>62.16</td>
    </tr>
    <tr>
        <td>HQQ</td>
        <td>60.02 </td>
        <td>75.41 </td>
        <td>60.79 </td>
        <td>74.11 </td>
        <td>81.01 </td>
        <td>27.29 </td>
        <td>32.60 </td>
        <td>82.97 </td>
        <td>66.79 </td>
        <td>79.92 </td>
        <td>49.32 </td>
        <td><b>62.75 </td>
    </tr>
    <tr>
        <td>Omniquant</td>
        <td>59.71 </td>
        <td>73.94 </td>
        <td>60.62 </td>
        <td>73.56 </td>
        <td>80.36 </td>
        <td>26.68 </td>
        <td>30.80 </td>
        <td>83.58 </td>
        <td>65.70 </td>
        <td>80.01 </td>
        <td>49.06 </td>
        <td>62.18 </td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>60.47</td>
        <td>75.59</td>
        <td>61.03</td>
        <td>73.88</td>
        <td>80.09</td>
        <td>27.54</td>
        <td>31.60</td>
        <td>83.09</td>
        <td>66.07</td>
        <td>79.97</td>
        <td>49.49</td>
        <td>62.62</td>
    </tr>
    <tr>
        <td rowspan="7">V2-7B</td>
        <td>FP16</td>
        <td>42.69</td>
        <td>73.90</td>
        <td>57.15</td>
        <td>68.90</td>
        <td>78.07</td>
        <td>25.21</td>
        <td>31.40</td>
        <td>77.74</td>
        <td>62.82</td>
        <td>76.35</td>
        <td>43.52</td>
        <td>57.98</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>40.91</td>
        <td>72.44</td>
        <td>56.91</td>
        <td>68.35</td>
        <td>77.58</td>
        <td>24.97</td>
        <td>31.20</td>
        <td>77.61</td>
        <td>56.32</td>
        <td>76.26</td>
        <td>43.52</td>
        <td>56.92</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>42.57</td>
        <td>73.28</td>
        <td>56.36</td>
        <td>69.06</td>
        <td>78.02</td>
        <td>25.34</td>
        <td>30.20</td>
        <td>75.72</td>
        <td>57.04</td>
        <td>75.63</td>
        <td>42.15</td>
        <td>56.85</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>41.00</td>
        <td>72.60</td>
        <td>56.40</td>
        <td>68.98</td>
        <td>77.31</td>
        <td>25.70</td>
        <td>31.60</td>
        <td>78.75</td>
        <td>58.48</td>
        <td>76.14</td>
        <td>43.86</td>
        <td>57.35</td>
    </tr>
        <tr>
        <td>HQQ</td>
        <td>41.79 </td>
        <td>73.20 </td>
        <td>56.21 </td>
        <td>68.43 </td>
        <td>77.58 </td>
        <td>25.83 </td>
        <td>31.60 </td>
        <td>76.09 </td>
        <td>62.82 </td>
        <td>75.84 </td>
        <td>42.15 </td>
        <td>57.41 </td>
    </tr>
    </tr>
        <tr>
        <td>Omniquant</td>
        <td>41.72 </td>
        <td>73.04 </td>
        <td>56.59 </td>
        <td>68.98 </td>
        <td>77.91 </td>
        <td>24.97 </td>
        <td>30.80 </td>
        <td>75.81 </td>
        <td>61.37 </td>
        <td>75.76 </td>
        <td>43.34 </td>
        <td>57.30 </td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>41.82</td>
        <td>72.75</td>
        <td>56.79</td>
        <td>68.67</td>
        <td>78.13</td>
        <td>25.58</td>
        <td>30.20</td>
        <td>77.49</td>
        <td>63.54</td>
        <td>75.76</td>
        <td>42.58</td>
        <td><b>57.57</td>
    </tr>
    <tr>
        <td rowspan="7">V2-13B</td>
        <td>FP16</td>
        <td>52.86</td>
        <td>76.77</td>
        <td>60.04</td>
        <td>72.14</td>
        <td>79.05</td>
        <td>25.95</td>
        <td>35.20</td>
        <td>80.55</td>
        <td>65.34</td>
        <td>79.38</td>
        <td>48.38</td>
        <td>61.42</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>52.10</td>
        <td>76.27</td>
        <td>59.77</td>
        <td>72.14</td>
        <td>78.62</td>
        <td>24.72</td>
        <td>34.20</td>
        <td>80.24</td>
        <td>62.09</td>
        <td>79.00</td>
        <td>47.95</td>
        <td>60.65</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>52.66</td>
        <td>76.54</td>
        <td>59.76</td>
        <td>72.14</td>
        <td>78.35</td>
        <td>25.70</td>
        <td>34.00</td>
        <td>79.33</td>
        <td>66.43</td>
        <td>78.58</td>
        <td>47.53</td>
        <td><b>61.00</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>52.39</td>
        <td>76.89</td>
        <td>59.97</td>
        <td>73.24</td>
        <td>79.00</td>
        <td>25.21</td>
        <td>32.60</td>
        <td>80.40</td>
        <td>63.54</td>
        <td>79.04</td>
        <td>47.70</td>
        <td>60.91</td>
    </tr>
    <tr>
        <td>HQQ</td>
        <td>52.09 </td>
        <td>75.74 </td>
        <td>59.46 </td>
        <td>72.14 </td>
        <td>78.45 </td>
        <td>24.36 </td>
        <td>33.60 </td>
        <td>79.17 </td>
        <td>66.06 </td>
        <td>79.00 </td>
        <td>47.01 </td>
        <td>60.65 </td>
    </tr>
    <tr>
        <td>Omniquant</td>
        <td>52.01 </td>
        <td>76.17 </td>
        <td>59.53 </td>
        <td>72.06 </td>
        <td>78.35 </td>
        <td>23.87 </td>
        <td>33.40 </td>
        <td>80.80 </td>
        <td>66.07 </td>
        <td>78.37 </td>
        <td>47.18 </td>
        <td>60.51 </td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>51.92</td>
        <td>76.46</td>
        <td>59.87</td>
        <td>71.67</td>
        <td>79.00</td>
        <td>25.83</td>
        <td>35.20</td>
        <td>79.60</td>
        <td>63.54</td>
        <td>79.25</td>
        <td>47.01</td>
        <td>60.85</td>
    </tr>
    <tr>
        <td rowspan="7">V2-70B</td>
        <td>FP16</td>
        <td>66.23</td>
        <td>79.64</td>
        <td>64.77</td>
        <td>77.98</td>
        <td>82.15</td>
        <td>30.60</td>
        <td>37.20</td>
        <td>83.70</td>
        <td>67.87</td>
        <td>82.70</td>
        <td>54.44</td>
        <td>66.12</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>64.91</td>
        <td>79.06</td>
        <td>63.93</td>
        <td>78.14</td>
        <td>81.66</td>
        <td>30.11</td>
        <td>37.00</td>
        <td>83.61</td>
        <td>68.59</td>
        <td>82.79</td>
        <td>54.78</td>
        <td>65.87</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>65.63</td>
        <td>79.22</td>
        <td>64.45</td>
        <td>78.22</td>
        <td>81.88</td>
        <td>31.09</td>
        <td>37.00</td>
        <td>84.19</td>
        <td>69.31</td>
        <td>82.79</td>
        <td>54.61</td>
        <td>66.22</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>65.79</td>
        <td>79.76</td>
        <td>64.48</td>
        <td>77.58</td>
        <td>82.32</td>
        <td>30.72</td>
        <td>38.00</td>
        <td>83.06</td>
        <td>68.95</td>
        <td>82.70</td>
        <td>55.12</td>
        <td>66.23</td>
    </tr>
    <tr>
        <td>HQQ</td>
        <td>65.34 </td>
        <td>79.14 </td>
        <td>64.56 </td>
        <td>77.35 </td>
        <td>81.56 </td>
        <td>30.48 </td>
        <td>37.20 </td>
        <td>83.67 </td>
        <td>69.31 </td>
        <td>82.83 </td>
        <td>55.20 </td>
        <td>66.06 </td>
    </tr>
    <tr>
        <td>Omniquant</td>
        <td>65.30 </td>
        <td>79.39 </td>
        <td>64.52 </td>
        <td>77.51 </td>
        <td>81.88 </td>
        <td>30.60 </td>
        <td>37.40 </td>
        <td>83.39 </td>
        <td>68.23 </td>
        <td>82.91 </td>
        <td>55.12 </td>
        <td>66.02 </td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>65.65</td>
        <td>79.49</td>
        <td>64.60</td>
        <td>78.30</td>
        <td>82.05</td>
        <td>31.58</td>
        <td>37.40</td>
        <td>84.83</td>
        <td>68.95</td>
        <td>82.87</td>
        <td>54.52</td>
        <td><b>66.39</td>
    </tr>
    <tr>
        <td rowspan="5">V1-7B</td>
        <td>FP16</td>
        <td>32.74</td>
        <td>73.53</td>
        <td>56.94</td>
        <td>70.01</td>
        <td>78.67</td>
        <td>22.03</td>
        <td>34.60</td>
        <td>75.08</td>
        <td>66.43</td>
        <td>75.25</td>
        <td>41.81</td>
        <td>57.01</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>32.63</td>
        <td>72.31</td>
        <td>56.26</td>
        <td>70.01</td>
        <td>78.45</td>
        <td>20.93</td>
        <td>33.60</td>
        <td>74.74</td>
        <td>64.26</td>
        <td>74.71</td>
        <td>42.75</td>
        <td>56.42</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>31.16</td>
        <td>72.40</td>
        <td>55.85</td>
        <td>70.09</td>
        <td>78.13</td>
        <td>22.28</td>
        <td>30.40</td>
        <td>74.65</td>
        <td>64.26</td>
        <td>74.20</td>
        <td>40.19</td>
        <td>55.78</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>33.42</td>
        <td>72.95</td>
        <td>56.30</td>
        <td>68.75</td>
        <td>77.97</td>
        <td>21.42</td>
        <td>32.80</td>
        <td>74.89</td>
        <td>62.09</td>
        <td>75.00</td>
        <td>41.21</td>
        <td>56.07</td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>32.15</td>
        <td>72.85</td>
        <td>56.45</td>
        <td>70.17</td>
        <td>78.51</td>
        <td>22.28</td>
        <td>32.80</td>
        <td>75.14</td>
        <td>67.87</td>
        <td>75.13</td>
        <td>41.89</td>
        <td><b>56.84</td>
    </tr>
    <tr>
        <td rowspan="5">V1-13B</td>
        <td>FP16</td>
        <td>44.21</td>
        <td>76.21</td>
        <td>59.92</td>
        <td>72.77</td>
        <td>79.16</td>
        <td>25.70</td>
        <td>33.20</td>
        <td>77.89</td>
        <td>70.76</td>
        <td>77.40</td>
        <td>46.42</td>
        <td>60.33</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>42.71</td>
        <td>75.26</td>
        <td>59.30</td>
        <td>72.53</td>
        <td>79.54</td>
        <td>25.95</td>
        <td>32.60</td>
        <td>76.76</td>
        <td>65.34</td>
        <td>76.98</td>
        <td>45.82</td>
        <td>59.34</td>
    </tr>
    <tr>
        <td>GPTQ*</td>
        <td>42.65</td>
        <td>75.41</td>
        <td>59.51</td>
        <td>72.93</td>
        <td>79.33</td>
        <td>24.97</td>
        <td>32.40</td>
        <td>77.49</td>
        <td>68.23</td>
        <td>76.89</td>
        <td>45.56</td>
        <td>59.58</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>42.66</td>
        <td>75.76</td>
        <td>59.50</td>
        <td>72.77</td>
        <td>78.89</td>
        <td>26.56</td>
        <td>33.60</td>
        <td>77.46</td>
        <td>68.59</td>
        <td>76.94</td>
        <td>45.48</td>
        <td>59.84</td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>42.27</td>
        <td>76.17</td>
        <td>59.53</td>
        <td>73.56</td>
        <td>79.33</td>
        <td>25.70</td>
        <td>32.80</td>
        <td>78.20</td>
        <td>70.04</td>
        <td>76.94</td>
        <td>46.25</td>
        <td><b>60.07</td>
    </tr>
    <tr>
        <td rowspan="5">V1-30B</td>
        <td>FP16</td>
        <td>55.14</td>
        <td>77.55</td>
        <td>63.33</td>
        <td>75.85</td>
        <td>81.12</td>
        <td>28.27</td>
        <td>36.00</td>
        <td>82.78</td>
        <td>66.79</td>
        <td>80.39</td>
        <td>52.90</td>
        <td>63.65</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>54.24</td>
        <td>77.02</td>
        <td>62.90</td>
        <td>74.35</td>
        <td>80.52</td>
        <td>27.29</td>
        <td>34.20</td>
        <td>81.96</td>
        <td>67.15</td>
        <td>80.89</td>
        <td>52.05</td>
        <td>62.96</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>54.20</td>
        <td>77.41</td>
        <td>62.79</td>
        <td>75.14</td>
        <td>80.41</td>
        <td>27.54</td>
        <td>34.60</td>
        <td>81.93</td>
        <td>67.51</td>
        <td>80.05</td>
        <td>50.51</td>
        <td>62.92</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>55.14</td>
        <td>77.49</td>
        <td>63.08</td>
        <td>75.77</td>
        <td>80.52</td>
        <td>27.29</td>
        <td>34.20</td>
        <td>82.87</td>
        <td>67.15</td>
        <td>80.43</td>
        <td>52.90</td>
        <td><b>63.35</td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>54.68</td>
        <td>77.90</td>
        <td>62.93</td>
        <td>74.82</td>
        <td>80.47</td>
        <td>28.15</td>
        <td>35.80</td>
        <td>82.39</td>
        <td>66.79</td>
        <td>80.13</td>
        <td>51.11</td>
        <td>63.20</td>
    </tr>
    <tr>
        <td rowspan="5">V1-65B</td>
        <td>FP16</td>
        <td>59.79</td>
        <td>79.12</td>
        <td>64.53</td>
        <td>77.35</td>
        <td>81.23</td>
        <td>27.91</td>
        <td>38.00</td>
        <td>84.86</td>
        <td>69.68</td>
        <td>81.36</td>
        <td>52.82</td>
        <td>65.15</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>59.53</td>
        <td>79.51</td>
        <td>64.63</td>
        <td>77.35</td>
        <td>80.96</td>
        <td>27.91</td>
        <td>38.40</td>
        <td>84.43</td>
        <td>71.48</td>
        <td>81.48</td>
        <td>52.22</td>
        <td><b>65.26</td>
    </tr>
    <tr>
        <td>GPTQ*</td>
        <td>60.47</td>
        <td>78.79</td>
        <td>64.45</td>
        <td>76.24</td>
        <td>81.18</td>
        <td>28.03</td>
        <td>37.40</td>
        <td>83.85</td>
        <td>68.95</td>
        <td>81.57</td>
        <td>53.07</td>
        <td>64.91</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>59.45</td>
        <td>79.31</td>
        <td>64.67</td>
        <td>76.72</td>
        <td>81.56</td>
        <td>28.15</td>
        <td>38.00</td>
        <td>84.43</td>
        <td>71.12</td>
        <td>81.10</td>
        <td>52.13</td>
        <td>65.15</td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>58.93</td>
        <td>79.22</td>
        <td>64.48</td>
        <td>77.03</td>
        <td>81.28</td>
        <td>27.91</td>
        <td>38.60</td>
        <td>84.31</td>
        <td>70.76</td>
        <td>81.19</td>
        <td>52.22</td>
        <td>65.08</td>
    </tr>
</table>

</br>

### 3. Accuracies $\uparrow$ across 11 tasks(0-shot) of LLaMA and Mistral models at W3G128.

<table border="1">
    <tr>
        <td></td>
        <td></td>
        <td>Mmlu</td>
        <td>Lamb.</td>
        <td>Hella.</td>
        <td>Wino.</td>
        <td>Piqa</td>
        <td>Truth.</td>
        <td>Open.</td>
        <td>Boolq</td>
        <td>RTE</td>
        <td>ARC-e</td>
        <td>ARC-c.</td>
        <td>Avg.</td>
    </tr>
    <tr>
        <td rowspan="7">Mistral-7B</td>
        <td>FP16</td>
        <td>61.35</td>
        <td>75.68</td>
        <td>61.27</td>
        <td>74.03</td>
        <td>80.79</td>
        <td>28.03</td>
        <td>32.80</td>
        <td>83.67</td>
        <td>67.51</td>
        <td>80.81</td>
        <td>50.34</td>
        <td>63.30</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>53.49</td>
        <td>68.74</td>
        <td>58.12</td>
        <td>68.27</td>
        <td>79.33</td>
        <td>24.60</td>
        <td>29.60</td>
        <td>79.97</td>
        <td>57.40</td>
        <td>76.89</td>
        <td>43.77</td>
        <td>58.20</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>55.84</td>
        <td>73.04</td>
        <td>57.61</td>
        <td>70.24</td>
        <td>78.67</td>
        <td>24.85</td>
        <td>30.80</td>
        <td>81.44</td>
        <td>63.54</td>
        <td>77.27</td>
        <td>45.65</td>
        <td>59.91</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>55.61</td>
        <td>73.69</td>
        <td>57.86</td>
        <td>71.27</td>
        <td>79.82</td>
        <td>26.07</td>
        <td>29.00</td>
        <td>81.10</td>
        <td>59.21</td>
        <td>79.00</td>
        <td>46.93</td>
        <td>59.96</td>
    </tr>
    <tr>
        <td>HQQ</td>
        <td>53.97 </td>
        <td>68.66 </td>
        <td>58.59 </td>
        <td>72.22 </td>
        <td>78.73 </td>
        <td>25.70 </td>
        <td>30.00 </td>
        <td>80.24 </td>
        <td>63.90 </td>
        <td>76.81 </td>
        <td>43.86 </td>
        <td>59.33 </td>
    </tr>
    <tr>
        <td>Omniquant</td>
        <td>54.79 </td>
        <td>69.34 </td>
        <td>58.42 </td>
        <td>68.51 </td>
        <td>79.38 </td>
        <td>24.85 </td>
        <td>28.80 </td>
        <td>80.15 </td>
        <td>56.68 </td>
        <td>77.74 </td>
        <td>45.14 </td>
        <td>58.53 </td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>57.54</td>
        <td>73.01</td>
        <td>59.60</td>
        <td>72.85</td>
        <td>79.54</td>
        <td>25.70</td>
        <td>31.60</td>
        <td>81.74</td>
        <td>58.12</td>
        <td>78.70</td>
        <td>46.33</td>
        <td><b>60.43</td>
    </tr>
    <tr>
        <td rowspan="7">V2-7B</td>
        <td>FP16</td>
        <td>42.69</td>
        <td>73.90</td>
        <td>57.15</td>
        <td>68.90</td>
        <td>78.07</td>
        <td>25.21</td>
        <td>31.40</td>
        <td>77.74</td>
        <td>62.82</td>
        <td>76.35</td>
        <td>43.52</td>
        <td>57.98</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>34.22</td>
        <td>65.96</td>
        <td>54.90</td>
        <td>67.56</td>
        <td>76.28</td>
        <td>24.48</td>
        <td>30.80</td>
        <td>71.68</td>
        <td>54.51</td>
        <td>72.98</td>
        <td>38.57</td>
        <td>53.81</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>36.11</td>
        <td>69.61</td>
        <td>53.66</td>
        <td>68.59</td>
        <td>76.01</td>
        <td>21.91</td>
        <td>27.80</td>
        <td>73.43</td>
        <td>54.51</td>
        <td>73.74</td>
        <td>40.19</td>
        <td>54.14</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>35.82</td>
        <td>69.90</td>
        <td>54.98</td>
        <td>67.40</td>
        <td>76.01</td>
        <td>25.21</td>
        <td>29.80</td>
        <td>74.68</td>
        <td>57.76</td>
        <td>74.07</td>
        <td>41.64</td>
        <td>55.21</td>
    </tr>
    <tr>
        <td>HQQ</td>
        <td>34.40 </td>
        <td>66.64 </td>
        <td>53.27 </td>
        <td>67.01 </td>
        <td>75.46 </td>
        <td>25.46 </td>
        <td>28.80 </td>
        <td>73.58 </td>
        <td>61.37 </td>
        <td>72.94 </td>
        <td>38.48 </td>
        <td>54.31 </td>
    </tr>
    <tr>
        <td>Omniquant</td>
        <td>34.51 </td>
        <td>69.75 </td>
        <td>54.42 </td>
        <td>66.69 </td>
        <td>76.77 </td>
        <td>24.24 </td>
        <td>31.40 </td>
        <td>73.21 </td>
        <td>56.68 </td>
        <td>74.37 </td>
        <td>39.85 </td>
        <td>54.72 </td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>40.13</td>
        <td>71.01</td>
        <td>55.33</td>
        <td>68.27</td>
        <td>76.82</td>
        <td>25.34</td>
        <td>32.80</td>
        <td>75.32</td>
        <td>60.29</td>
        <td>75.25</td>
        <td>42.92</td>
        <td><b>56.68</td>
    </tr>
    <tr>
        <td rowspan="7">V2-13B</td>
        <td>FP16</td>
        <td>52.86</td>
        <td>76.77</td>
        <td>60.04</td>
        <td>72.14</td>
        <td>79.05</td>
        <td>25.95</td>
        <td>35.20</td>
        <td>80.55</td>
        <td>65.34</td>
        <td>79.38</td>
        <td>48.38</td>
        <td>61.42</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>48.01</td>
        <td>72.33</td>
        <td>57.74</td>
        <td>70.72</td>
        <td>78.07</td>
        <td>25.21</td>
        <td>32.00</td>
        <td>77.28</td>
        <td>60.65</td>
        <td>77.69</td>
        <td>44.62</td>
        <td>58.57</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>49.56</td>
        <td>75.24</td>
        <td>57.83</td>
        <td>70.88</td>
        <td>78.56</td>
        <td>24.97</td>
        <td>33.40</td>
        <td>78.44</td>
        <td>62.82</td>
        <td>77.99</td>
        <td>45.65</td>
        <td><b>59.58</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>49.77</td>
        <td>75.22</td>
        <td>58.58</td>
        <td>71.82</td>
        <td>77.75</td>
        <td>24.11</td>
        <td>34.20</td>
        <td>79.97</td>
        <td>53.43</td>
        <td>77.95</td>
        <td>44.62</td>
        <td>58.86</td>
    </tr>
    <tr>
        <td>HQQ</td>
        <td>48.40 </td>
        <td>73.22 </td>
        <td>57.66 </td>
        <td>69.77 </td>
        <td>77.31 </td>
        <td>24.11 </td>
        <td>30.60 </td>
        <td>76.97 </td>
        <td>60.29 </td>
        <td>77.15 </td>
        <td>43.60 </td>
        <td>58.10 </td>
    </tr>
    <tr>
        <td>Omniquant</td>
        <td>47.25 </td>
        <td>73.67 </td>
        <td>58.46 </td>
        <td>70.01 </td>
        <td>78.40 </td>
        <td>24.36 </td>
        <td>33.60 </td>
        <td>79.79 </td>
        <td>64.62 </td>
        <td>77.86 </td>
        <td>46.16 </td>
        <td>59.18 </td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>49.64</td>
        <td>75.20</td>
        <td>59.11</td>
        <td>71.59</td>
        <td>78.29</td>
        <td>24.85</td>
        <td>34.20</td>
        <td>78.47</td>
        <td>58.12</td>
        <td>78.58</td>
        <td>45.82</td>
        <td>59.44</td>
    </tr>
    <tr>
        <td rowspan="7">V2-70B</td>
        <td>FP16</td>
        <td>66.23</td>
        <td>79.64</td>
        <td>64.77</td>
        <td>77.98</td>
        <td>82.15</td>
        <td>30.60</td>
        <td>37.20</td>
        <td>83.70</td>
        <td>67.87</td>
        <td>82.70</td>
        <td>54.44</td>
        <td>66.12</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>61.15</td>
        <td>77.95</td>
        <td>61.98</td>
        <td>77.90</td>
        <td>80.79</td>
        <td>29.74</td>
        <td>36.00</td>
        <td>81.28</td>
        <td>64.62</td>
        <td>81.10</td>
        <td>52.39</td>
        <td>64.08</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>63.15</td>
        <td>79.06</td>
        <td>62.94</td>
        <td>77.66</td>
        <td>81.45</td>
        <td>30.72</td>
        <td>36.20</td>
        <td>81.53</td>
        <td>67.87</td>
        <td>81.65</td>
        <td>53.67</td>
        <td>65.08</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>64.09</td>
        <td>79.47</td>
        <td>63.75</td>
        <td>76.48</td>
        <td>81.77</td>
        <td>29.74</td>
        <td>37.20</td>
        <td>82.69</td>
        <td>66.06</td>
        <td>81.40</td>
        <td>53.67</td>
        <td>65.12</td>
    </tr>
    <tr>
        <td>HQQ</td>
        <td>63.45 </td>
        <td>78.05 </td>
        <td>63.12 </td>
        <td>77.03 </td>
        <td>81.01 </td>
        <td>29.38 </td>
        <td>36.60 </td>
        <td>82.23 </td>
        <td>66.43 </td>
        <td>81.78 </td>
        <td>53.67 </td>
        <td>64.80 </td>
    </tr>
    <tr>
        <td>Omniquant</td>
        <td>63.18 </td>
        <td>78.63 </td>
        <td>63.54 </td>
        <td>76.48 </td>
        <td>81.50 </td>
        <td>30.35 </td>
        <td>35.80 </td>
        <td>82.57 </td>
        <td>70.40 </td>
        <td>81.02 </td>
        <td>52.82 </td>
        <td>65.12 </td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>64.94</td>
        <td>78.89</td>
        <td>63.83</td>
        <td>76.56</td>
        <td>81.50</td>
        <td>31.21</td>
        <td>37.20</td>
        <td>81.41</td>
        <td>68.59</td>
        <td>81.73</td>
        <td>52.56</td>
        <td><b>65.31</td>
    </tr>
    <tr>
        <td rowspan="5">V1-7B</td>
        <td>FP16</td>
        <td>32.74</td>
        <td>73.53</td>
        <td>56.94</td>
        <td>70.01</td>
        <td>78.67</td>
        <td>22.03</td>
        <td>34.60</td>
        <td>75.08</td>
        <td>66.43</td>
        <td>75.25</td>
        <td>41.81</td>
        <td>57.01</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>28.00</td>
        <td>67.67</td>
        <td>53.43</td>
        <td>66.38</td>
        <td>76.50</td>
        <td>21.42</td>
        <td>31.20</td>
        <td>72.72</td>
        <td>59.21</td>
        <td>70.92</td>
        <td>38.31</td>
        <td>53.25</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>30.16</td>
        <td>66.31</td>
        <td>53.92</td>
        <td>67.48</td>
        <td>76.82</td>
        <td>21.42</td>
        <td>29.60</td>
        <td>71.31</td>
        <td>59.21</td>
        <td>72.22</td>
        <td>38.74</td>
        <td>53.38</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>30.33</td>
        <td>70.19</td>
        <td>54.53</td>
        <td>68.98</td>
        <td>76.71</td>
        <td>20.81</td>
        <td>31.60</td>
        <td>74.68</td>
        <td>64.62</td>
        <td>73.23</td>
        <td>38.91</td>
        <td><b>54.96</td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>25.85</td>
        <td>70.95</td>
        <td>55.45</td>
        <td>69.69</td>
        <td>77.37</td>
        <td>21.66</td>
        <td>32.00</td>
        <td>73.88</td>
        <td>60.29</td>
        <td>73.48</td>
        <td>39.33</td>
        <td>54.54</td>
    </tr>
    <tr>
        <td rowspan="5">V1-13B</td>
        <td>FP16</td>
        <td>44.21</td>
        <td>76.21</td>
        <td>59.92</td>
        <td>72.77</td>
        <td>79.16</td>
        <td>25.70</td>
        <td>33.20</td>
        <td>77.89</td>
        <td>70.76</td>
        <td>77.40</td>
        <td>46.42</td>
        <td>60.33</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>34.87</td>
        <td>69.65</td>
        <td>57.25</td>
        <td>70.48</td>
        <td>77.31</td>
        <td>26.93</td>
        <td>32.00</td>
        <td>71.44</td>
        <td>62.82</td>
        <td>75.63</td>
        <td>43.94</td>
        <td>56.57</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>35.51</td>
        <td>73.08</td>
        <td>57.89</td>
        <td>70.80</td>
        <td>77.37</td>
        <td>24.48</td>
        <td>31.40</td>
        <td>77.52</td>
        <td>62.82</td>
        <td>74.41</td>
        <td>43.26</td>
        <td>57.14</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>40.53</td>
        <td>73.94</td>
        <td>57.89</td>
        <td>69.53</td>
        <td>78.94</td>
        <td>26.68</td>
        <td>33.40</td>
        <td>74.83</td>
        <td>65.34</td>
        <td>75.93</td>
        <td>45.05</td>
        <td>58.37</td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>39.16</td>
        <td>75.22</td>
        <td>58.64</td>
        <td>71.59</td>
        <td>78.94</td>
        <td>25.95</td>
        <td>35.20</td>
        <td>76.30</td>
        <td>65.34</td>
        <td>76.52</td>
        <td>45.39</td>
        <td><b>58.93</td>
    </tr>
    <tr>
        <td rowspan="5">V1-30B</td>
        <td>FP16</td>
        <td>55.14</td>
        <td>77.55</td>
        <td>63.33</td>
        <td>75.85</td>
        <td>81.12</td>
        <td>28.27</td>
        <td>36.00</td>
        <td>82.78</td>
        <td>66.79</td>
        <td>80.39</td>
        <td>52.90</td>
        <td>63.65</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>52.41</td>
        <td>75.08</td>
        <td>61.45</td>
        <td>74.27</td>
        <td>79.87</td>
        <td>25.95</td>
        <td>33.00</td>
        <td>81.38</td>
        <td>65.34</td>
        <td>79.12</td>
        <td>48.89</td>
        <td>61.52</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>51.39</td>
        <td>74.97</td>
        <td>60.35</td>
        <td>75.30</td>
        <td>79.60</td>
        <td>26.93</td>
        <td>34.80</td>
        <td>82.75</td>
        <td>64.62</td>
        <td>78.11</td>
        <td>48.46</td>
        <td>61.57</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>53.84</td>
        <td>76.71</td>
        <td>61.94</td>
        <td>75.14</td>
        <td>80.03</td>
        <td>25.34</td>
        <td>34.40</td>
        <td>81.90</td>
        <td>67.15</td>
        <td>79.59</td>
        <td>50.77</td>
        <td>62.44</td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>54.39</td>
        <td>77.49</td>
        <td>62.13</td>
        <td>74.03</td>
        <td>80.47</td>
        <td>27.30</td>
        <td>35.00</td>
        <td>79.76</td>
        <td>68.59</td>
        <td>79.46</td>
        <td>48.98</td>
        <td><b>62.51</td>
    </tr>
    <tr>
        <td rowspan="5">V1-65B</td>
        <td>FP16</td>
        <td>59.79</td>
        <td>79.12</td>
        <td>64.53</td>
        <td>77.35</td>
        <td>81.23</td>
        <td>27.91</td>
        <td>38.00</td>
        <td>84.86</td>
        <td>69.68</td>
        <td>81.36</td>
        <td>52.82</td>
        <td>65.15</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>57.47</td>
        <td>77.43</td>
        <td>63.23</td>
        <td>75.93</td>
        <td>80.41</td>
        <td>28.64</td>
        <td>38.40</td>
        <td>82.69</td>
        <td>66.43</td>
        <td>80.22</td>
        <td>51.19</td>
        <td>63.82</td>
    </tr>
    <tr>
        <td>GPTQ*</td>
        <td>57.92</td>
        <td>78.69</td>
        <td>62.98</td>
        <td>76.87</td>
        <td>80.63</td>
        <td>27.66</td>
        <td>37.60</td>
        <td>84.16</td>
        <td>68.95</td>
        <td>80.89</td>
        <td>51.19</td>
        <td>64.32</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>58.87</td>
        <td>77.94</td>
        <td>63.77</td>
        <td>75.37</td>
        <td>80.96</td>
        <td>27.66</td>
        <td>36.80</td>
        <td>85.02</td>
        <td>71.12</td>
        <td>81.10</td>
        <td>50.34</td>
        <td>64.45</td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>58.30</td>
        <td>78.11</td>
        <td>63.60</td>
        <td>76.56</td>
        <td>80.85</td>
        <td>29.50</td>
        <td>37.80</td>
        <td>84.80</td>
        <td>70.04</td>
        <td>80.22</td>
        <td>50.68</td>
        <td><b>64.59</td>
    </tr>
</table>

</br>

### 4. Accuracies $\uparrow$ across 11 tasks(0-shot) of LLaMA and Mistral models at W2G128.

<table border="1">
    <tr>
        <td></td>
        <td></td>
        <td>Mmlu</td>
        <td>Lamb.</td>
        <td>Hella.</td>
        <td>Wino.</td>
        <td>Piqa</td>
        <td>Truth.</td>
        <td>Open.</td>
        <td>Boolq</td>
        <td>RTE</td>
        <td>ARC-e</td>
        <td>ARC-c.</td>
        <td>Avg.</td>
    </tr>
    <tr>
        <td rowspan="7">Mistral-7B</td>
        <td>FP16</td>
        <td>61.35</td>
        <td>75.68</td>
        <td>61.27</td>
        <td>74.03</td>
        <td>80.79</td>
        <td>28.03</td>
        <td>32.80</td>
        <td>83.67</td>
        <td>67.51</td>
        <td>80.81</td>
        <td>50.34</td>
        <td>63.30</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>23.45</td>
        <td>0.14</td>
        <td>27.43</td>
        <td>49.64</td>
        <td>54.30</td>
        <td>24.24</td>
        <td>15.20</td>
        <td>38.69</td>
        <td>51.99</td>
        <td>29.08</td>
        <td>21.59</td>
        <td>30.52</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>25.23</td>
        <td>30.47</td>
        <td>38.28</td>
        <td>53.83</td>
        <td>64.91</td>
        <td>24.11</td>
        <td>17.40</td>
        <td>58.29</td>
        <td>50.90</td>
        <td>47.77</td>
        <td>24.57</td>
        <td>39.61</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>25.38</td>
        <td>0.00</td>
        <td>25.71</td>
        <td>52.01</td>
        <td>51.58</td>
        <td>23.99</td>
        <td>17.60</td>
        <td>37.83</td>
        <td>47.29</td>
        <td>26.98</td>
        <td>22.27</td>
        <td>30.06</td>
    </tr>
    <tr>
        <td>HQQ</td>
        <td>23.35 </td>
        <td>0.85 </td>
        <td>27.77 </td>
        <td>51.62 </td>
        <td>56.69 </td>
        <td>26.68 </td>
        <td>15.80 </td>
        <td>40.55 </td>
        <td>53.43 </td>
        <td>28.62 </td>
        <td>20.14 </td>
        <td>31.41 </td>
    </tr>
    <tr>
        <td>Omniquant</td>
        <td>23.24 </td>
        <td>5.38 </td>
        <td>29.38 </td>
        <td>49.72 </td>
        <td>56.09 </td>
        <td>26.32 </td>
        <td>16.60 </td>
        <td>41.99 </td>
        <td>52.71 </td>
        <td>32.11 </td>
        <td>20.39 </td>
        <td>32.17 </td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>40.46</td>
        <td>58.61</td>
        <td>50.87</td>
        <td>62.90</td>
        <td>75.84</td>
        <td>24.85</td>
        <td>22.80</td>
        <td>78.56</td>
        <td>57.04</td>
        <td>70.88</td>
        <td>37.03</td>
        <td><b>52.71 </td>
    </tr>
    <tr>
        <td rowspan="7">V2-7B</td>
        <td>FP16</td>
        <td>42.69</td>
        <td>73.90</td>
        <td>57.15</td>
        <td>68.90</td>
        <td>78.07</td>
        <td>25.21</td>
        <td>31.40</td>
        <td>77.74</td>
        <td>62.82</td>
        <td>76.35</td>
        <td>43.52</td>
        <td>57.98</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>23.98</td>
        <td>0.02</td>
        <td>26.04</td>
        <td>49.49</td>
        <td>52.50</td>
        <td>24.85</td>
        <td>15.20</td>
        <td>41.01</td>
        <td>49.10</td>
        <td>27.48</td>
        <td>19.71</td>
        <td>29.94</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>23.65</td>
        <td>11.72</td>
        <td>32.59</td>
        <td>55.17</td>
        <td>58.32</td>
        <td>25.95</td>
        <td>15.80</td>
        <td>52.14</td>
        <td>51.99</td>
        <td>40.45</td>
        <td>21.25</td>
        <td>35.37</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>25.38</td>
        <td>0.00</td>
        <td>25.69</td>
        <td>49.96</td>
        <td>52.34</td>
        <td>23.75</td>
        <td>17.80</td>
        <td>37.83</td>
        <td>52.71</td>
        <td>24.62</td>
        <td>21.08</td>
        <td>30.10</td>
    </tr>
    <tr>
        <td>HQQ</td>
        <td>24.51 </td>
        <td>0.02 </td>
        <td>26.06 </td>
        <td>49.49 </td>
        <td>53.26 </td>
        <td>24.72 </td>
        <td>13.80 </td>
        <td>37.92 </td>
        <td>50.90 </td>
        <td>26.52 </td>
        <td>21.33 </td>
        <td>29.87 </td>
    </tr>
     <tr>
        <td>Omniquant</td>
        <td>22.97 </td>
        <td>35.53 </td>
        <td>40.28 </td>
        <td>55.88 </td>
        <td>65.13 </td>
        <td>22.89 </td>
        <td>15.60 </td>
        <td>63.24 </td>
        <td>53.07 </td>
        <td>50.13 </td>
        <td>23.46 </td>
        <td>40.74 </td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>27.20</td>
        <td>55.25</td>
        <td>47.35</td>
        <td>61.01</td>
        <td>72.96</td>
        <td>24.85</td>
        <td>25.60</td>
        <td>68.07</td>
        <td>54.51</td>
        <td>65.99</td>
        <td>32.25</td>
        <td><b>48.64</td>
    </tr>
    <tr>
        <td rowspan="7">V2-13B</td>
        <td>FP16</td>
        <td>52.86</td>
        <td>76.77</td>
        <td>60.04</td>
        <td>72.14</td>
        <td>79.05</td>
        <td>25.95</td>
        <td>35.20</td>
        <td>80.55</td>
        <td>65.34</td>
        <td>79.38</td>
        <td>48.38</td>
        <td>61.42</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>23.77</td>
        <td>7.47</td>
        <td>33.08</td>
        <td>49.01</td>
        <td>57.94</td>
        <td>26.19</td>
        <td>16.00</td>
        <td>47.74</td>
        <td>53.43</td>
        <td>32.03</td>
        <td>21.93</td>
        <td>33.51</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>24.69</td>
        <td>45.20</td>
        <td>41.06</td>
        <td>55.80</td>
        <td>67.08</td>
        <td>23.26</td>
        <td>19.80</td>
        <td>54.40</td>
        <td>52.35</td>
        <td>55.60</td>
        <td>27.82</td>
        <td>42.46</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>27.04</td>
        <td>0.00</td>
        <td>25.80</td>
        <td>51.85</td>
        <td>52.99</td>
        <td>23.62</td>
        <td>13.60</td>
        <td>62.17</td>
        <td>47.29</td>
        <td>26.22</td>
        <td>23.12</td>
        <td>32.16</td>
    </tr>
    <tr>
        <td>HQQ</td>
        <td>23.48 </td>
        <td>8.17 </td>
        <td>31.27 </td>
        <td>52.17 </td>
        <td>61.86 </td>
        <td>24.85 </td>
        <td>17.20 </td>
        <td>50.46 </td>
        <td>54.51 </td>
        <td>42.85 </td>
        <td>21.25 </td>
        <td>35.28 </td>
    </tr>
    <tr>
        <td>Omniquant</td>
        <td>25.53 </td>
        <td>49.84 </td>
        <td>46.23 </td>
        <td>57.93 </td>
        <td>70.13 </td>
        <td>24.60 </td>
        <td>21.80 </td>
        <td>66.85 </td>
        <td>55.60 </td>
        <td>63.22 </td>
        <td>30.29 </td>
        <td>46.55 </td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>34.33</td>
        <td>63.92</td>
        <td>53.35</td>
        <td>64.33</td>
        <td>76.17</td>
        <td>25.70</td>
        <td>26.00</td>
        <td>72.75</td>
        <td>61.73</td>
        <td>71.17</td>
        <td>38.57</td>
        <td><b>53.46</td>
    </tr>
    <tr>
        <td rowspan="7">V2-70B</td>
        <td>FP16</td>
        <td>66.23</td>
        <td>79.64</td>
        <td>64.77</td>
        <td>77.98</td>
        <td>82.15</td>
        <td>30.60</td>
        <td>37.20</td>
        <td>83.70</td>
        <td>67.87</td>
        <td>82.70</td>
        <td>54.44</td>
        <td>66.12</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>24.20</td>
        <td>20.18</td>
        <td>40.88</td>
        <td>54.85</td>
        <td>63.87</td>
        <td>24.11</td>
        <td>17.60</td>
        <td>43.06</td>
        <td>53.07</td>
        <td>50.51</td>
        <td>27.22</td>
        <td>38.14</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>23.12</td>
        <td>0.00</td>
        <td>25.04</td>
        <td>49.57</td>
        <td>49.51</td>
        <td>0.00</td>
        <td>27.60</td>
        <td>37.83</td>
        <td>52.71</td>
        <td>25.08</td>
        <td>22.70</td>
        <td>28.47</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>24.46</td>
        <td>0.00</td>
        <td>25.46</td>
        <td>51.38</td>
        <td>52.50</td>
        <td>23.50</td>
        <td>14.20</td>
        <td>62.17</td>
        <td>52.71</td>
        <td>25.76</td>
        <td>22.35</td>
        <td>32.23</td>
    </tr>
    <tr>
        <td>HQQ</td>
        <td>23.16 </td>
        <td>19.46 </td>
        <td>35.45 </td>
        <td>56.67 </td>
        <td>66.00 </td>
        <td>22.52 </td>
        <td>20.00 </td>
        <td>40.46 </td>
        <td>52.71 </td>
        <td>52.06 </td>
        <td>23.12 </td>
        <td>37.42 </td>
    </tr>
    <tr>
        <td>Omniquant</td>
        <td>33.84 </td>
        <td>61.83 </td>
        <td>52.44 </td>
        <td>64.33 </td>
        <td>74.10 </td>
        <td>24.48 </td>
        <td>28.20 </td>
        <td>71.68 </td>
        <td>53.07 </td>
        <td>67.21 </td>
        <td>33.28 </td>
        <td>51.31 </td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>54.04</td>
        <td>72.97</td>
        <td>59.65</td>
        <td>74.90</td>
        <td>79.00</td>
        <td>29.01</td>
        <td>34.80 </td>
        <td>79.63 </td>
        <td>69.68</td>
        <td>78.37 </td>
        <td> 46.59</td>
        <td><b>61.69</td>
    </tr>
    <tr>
        <td rowspan="5">V1-7B</td>
        <td>FP16</td>
        <td>32.74</td>
        <td>73.53</td>
        <td>56.94</td>
        <td>70.01</td>
        <td>78.67</td>
        <td>22.03</td>
        <td>34.60</td>
        <td>75.08</td>
        <td>66.43</td>
        <td>75.25</td>
        <td>41.81</td>
        <td>57.01</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>24.36</td>
        <td>0.52</td>
        <td>27.24</td>
        <td>49.25</td>
        <td>54.24</td>
        <td>24.24</td>
        <td>15.20</td>
        <td>39.63</td>
        <td>57.40</td>
        <td>27.86</td>
        <td>21.84</td>
        <td>31.07</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>22.95</td>
        <td>12.75</td>
        <td>33.36</td>
        <td>51.70</td>
        <td>60.07</td>
        <td>23.99</td>
        <td>13.40</td>
        <td>48.62</td>
        <td>53.07</td>
        <td>40.82</td>
        <td>21.50</td>
        <td>34.75</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>23.12</td>
        <td>0.00</td>
        <td>25.37</td>
        <td>53.28</td>
        <td>52.56</td>
        <td>25.21</td>
        <td>13.80</td>
        <td>37.83</td>
        <td>52.71</td>
        <td>25.63</td>
        <td>22.53</td>
        <td>30.18</td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>24.46</td>
        <td>13.53</td>
        <td>42.16</td>
        <td>56.99</td>
        <td>70.02</td>
        <td>24.60</td>
        <td>25.20</td>
        <td>62.91</td>
        <td>47.29</td>
        <td>60.90</td>
        <td>31.74</td>
        <td><b>41.80</td>
    </tr>
    <tr>
        <td rowspan="5">V1-13B</td>
        <td>FP16</td>
        <td>44.21</td>
        <td>76.21</td>
        <td>59.92</td>
        <td>72.77</td>
        <td>79.16</td>
        <td>25.70</td>
        <td>33.20</td>
        <td>77.89</td>
        <td>70.76</td>
        <td>77.40</td>
        <td>46.42</td>
        <td>60.33</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>24.66</td>
        <td>4.97</td>
        <td>29.67</td>
        <td>49.33</td>
        <td>57.24</td>
        <td>25.58</td>
        <td>12.40</td>
        <td>44.10</td>
        <td>53.79</td>
        <td>32.07</td>
        <td>22.01</td>
        <td>32.35</td>
    </tr>
    <tr>
        <td>GPTQ*</td>
        <td>26.43</td>
        <td>40.48</td>
        <td>39.47</td>
        <td>58.25</td>
        <td>66.97</td>
        <td>23.50</td>
        <td>18.60</td>
        <td>52.78</td>
        <td>50.54</td>
        <td>51.52</td>
        <td>25.00</td>
        <td>41.23</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>27.04</td>
        <td>0.00</td>
        <td>25.59</td>
        <td>50.36</td>
        <td>53.05</td>
        <td>24.11</td>
        <td>15.60</td>
        <td>62.17</td>
        <td>47.29</td>
        <td>25.97</td>
        <td>23.21</td>
        <td>32.22</td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>31.87</td>
        <td>59.65</td>
        <td>51.25</td>
        <td>67.64</td>
        <td>76.28</td>
        <td>25.58</td>
        <td>27.80</td>
        <td>69.11</td>
        <td>58.48</td>
        <td>70.71</td>
        <td>37.12</td>
        <td><b>52.32</td>
    </tr>
    <tr>
        <td rowspan="5">V1-30B</td>
        <td>FP16</td>
        <td>55.14</td>
        <td>77.55</td>
        <td>63.33</td>
        <td>75.85</td>
        <td>81.12</td>
        <td>28.27</td>
        <td>36.00</td>
        <td>82.78</td>
        <td>66.79</td>
        <td>80.39</td>
        <td>52.90</td>
        <td>63.65</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>23.24</td>
        <td>5.55</td>
        <td>27.22</td>
        <td>53.99</td>
        <td>56.80</td>
        <td>21.79</td>
        <td>18.20</td>
        <td>51.65</td>
        <td>53.07</td>
        <td>36.74</td>
        <td>21.33</td>
        <td>33.60</td>
    </tr>
    <tr>
        <td>GPTQ</td>
        <td>30.47</td>
        <td>49.93</td>
        <td>45.05</td>
        <td>61.88</td>
        <td>68.88</td>
        <td>23.26</td>
        <td>22.60</td>
        <td>68.29</td>
        <td>51.99</td>
        <td>60.69</td>
        <td>30.72</td>
        <td>46.70</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>27.04</td>
        <td>0.00</td>
        <td>25.41</td>
        <td>50.20</td>
        <td>52.94</td>
        <td>24.48</td>
        <td>16.60</td>
        <td>62.17</td>
        <td>47.29</td>
        <td>24.71</td>
        <td>23.38</td>
        <td>32.20</td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>40.83</td>
        <td>67.92</td>
        <td>56.73</td>
        <td>68.90</td>
        <td>76.17</td>
        <td>24.36</td>
        <td>31.60</td>
        <td>75.54</td>
        <td>62.45</td>
        <td>74.92</td>
        <td>42.41</td>
        <td><b>56.53</td>
    </tr>
    <tr>
        <td rowspan="5">V1-65B</td>
        <td>FP16</td>
        <td>59.79</td>
        <td>79.12</td>
        <td>64.53</td>
        <td>77.35</td>
        <td>81.23</td>
        <td>27.91</td>
        <td>38.00</td>
        <td>84.86</td>
        <td>69.68</td>
        <td>81.36</td>
        <td>52.82</td>
        <td>65.15</td>
    </tr>
    <tr>
        <td>RTN</td>
        <td>24.48</td>
        <td>32.78</td>
        <td>43.59</td>
        <td>57.85</td>
        <td>67.52</td>
        <td>22.89</td>
        <td>22.80</td>
        <td>61.53</td>
        <td>50.54</td>
        <td>52.10</td>
        <td>28.24</td>
        <td>42.21</td>
    </tr>
    <tr>
        <td>GPTQ*</td>
        <td>37.06</td>
        <td>67.44</td>
        <td>53.97</td>
        <td>69.46</td>
        <td>76.44</td>
        <td>24.36</td>
        <td>28.00</td>
        <td>73.64</td>
        <td>60.29</td>
        <td>71.34</td>
        <td>38.57</td>
        <td>54.60</td>
    </tr>
    <tr>
        <td>AWQ</td>
        <td>25.38</td>
        <td>0.00</td>
        <td>25.58</td>
        <td>49.96</td>
        <td>53.10</td>
        <td>24.24</td>
        <td>11.00</td>
        <td>37.83</td>
        <td>52.71</td>
        <td>24.96</td>
        <td>22.44</td>
        <td>29.75</td>
    </tr>
    <tr>
        <td>Ours</td>
        <td>47.21</td>
        <td>72.07</td>
        <td>60.06</td>
        <td> 73.24</td>
        <td>78.62</td>
        <td>25.46</td>
        <td>34.20</td>
        <td>80.64</td>
        <td>62.82</td>
        <td>77.48</td>
        <td>46.76</td>
        <td><b>59.87</td>
    </tr>
</table>


## Other data W4G128
<table border="1">
  <tr>
    <th>Model</th>
    <th>Method </th>
    <th>Acc AVG.</th>
    <th>MMLU</th>
    <th>Lamb.</th>
    <th>Hella.</th>
    <th>Wino.</th>
    <th>Piqa</th>
    <th>Truth.</th>
    <th>Open.</th>
    <th>Boolq</th>
    <th>RTE</th>
    <th>ARC-e</th>
    <th>ARC-c.</th>
    <th>wikitext2 ppl
    <th>ptb_new ppl</th>
    <th>c4_new ppl</th>
    <th>lm_eval wikitext ppl</th>
   
  </tr>

  <tr>
    <td rowspan="3">Intel/neural-chat-7b-v3-3 </td>
    <th>FP16</th>
    <td>67.92</td> <! acc avg -->
    <td>61.13</td> <! MMLU -->
    <td>73.03</td> <! Lambada_openai -->
    <td>66.39</td> <! Hellsaswag -->
    <td>76.40</td> <! Winogrande -->
    <td>81.01</td> <! Piqa -->
    <td>47.37</td> <! Truthfulqa -->
    <td>38.8</td> <! Openbookqa -->
    <td>86.97</td> <! Boolq -->
    <td>75.81</td> <! RTE -->
    <td>82.66</td> <! Arc easy -->
    <td>57.51</td> <! Arc Challenge  -->
    <td>6.00</td>  <! wikitext2 ppl  -->
    <td>48.96</td> <! ptb_new ppl  -->
    <td>9.65</td>    <! c4_new ppl  -->
    <td>-</td> <! lm-eval wikitext ppl  -->
  </tr>

  </tr>
    <th>Ours  </th>
    <td>66.90</td> <! acc avg -->
    <td>60.56</td> <! MMLU -->
    <td>72.19</td> <! Lambada_openai -->
    <td>65.28</td> <! Hellsaswag -->
    <td>75.37</td> <! Winogrande -->
    <td>81.18</td> <! Piqa -->
    <td>46.76</td> <! Truthfulqa -->
    <td>36.0</td> <! Openbookqa -->
    <td>86.91</td> <! Boolq -->
    <td>73.29</td> <! RTE -->
    <td>81.73</td> <! Arc easy -->
    <td>56.66</td> <! Arc Challenge  -->
    <td>6.21</td>  <! wikitext2 ppl  -->
    <td>59.78</td> <! ptb_new ppl  -->
    <td>10.01</td>    <! c4_new ppl  -->
    <td>-</td> <! lm-eval wikitext ppl  -->
  </tr>

  </tr>
    <th>Ours iters=1K,disable_quanted_input, minmax_lr=0.002</th>
    <td>67.70</td> <! acc avg -->
    <td>60.57</td> <! MMLU -->
    <td>73.74</td> <! Lambada_openai -->
    <td>65.62</td> <! Hellsaswag -->
    <td>77.43</td> <! Winogrande -->
    <td>80.85</td> <! Piqa -->
    <td>47.61</td> <! Truthfulqa -->
    <td>36.8</td> <! Openbookqa -->
    <td>86.94</td> <! Boolq -->
    <td>75.09</td> <! RTE -->
    <td>82.66</td> <! Arc easy -->
    <td>57.34</td> <! Arc Challenge  -->
    <td>6.17</td>  <! wikitext2 ppl  -->
    <td>59.12</td> <! ptb_new ppl  -->
    <td>9.83</td>    <! c4_new ppl  -->
    <td>-</td> <! lm-eval wikitext ppl  -->
  </tr>

  <tr>
    <td rowspan="3">mistralai/Mixtral-8x7B-v0.1 </td>
    <th>BF16</th>
   <td>67.16</td>
    <td>69.83</td>
    <td>78.44</td>
    <td>64.89</td>
    <td>76.40</td>
    <td>82.43</td>
    <td>34.15</td>
    <td>35.40</td>
    <td>84.98</td>
    <td>71.12</td>
    <td>84.22</td>
    <td>56.91</td>
    <td>3.84</td>
    <td>19.22</td>
    <td>7.41</td>
    <td>-</td>
 
  </tr>
  <tr>
    <th>Ours</th>
    <td>65.98</td>
    <td>68.90</td>
    <td>78.11</td>
    <td>64.31</td>
    <td>74.27</td>
    <td>82.10</td>
    <td>30.97</td>
    <td>34.20</td>
    <td>84.57</td>
    <td>67.87</td>
    <td>83.96</td>
    <td>56.57</td>
    <td>4.08</td>
    <td>354</td>
    <td>7.56</td>
    <td>-</td>
  </tr>
  <tr>
    <th>Ours iters=1K,disable_quanted_input 
    <td>66.78</td>
    <td>68.68</td>
    <td>78.61</td>
    <td>64.40</td>
    <td>76.56</td>
    <td>81.99</td>
    <td>32.56</td>
    <td>34.80</td>
    <td>85.96</td>
    <td>70.76</td>
    <td>83.96</td>
    <td>56.31</td>
    <td>3.99</td>
    <td>17.65</td>
    <td>7.52</td>
    <td>-</td>
 
  </tr>
  <tr>
    <td rowspan="3">microsoft/phi-2 </td>
    <th>FP16</th>
    <td>61.80</td>
    <td>56.40</td>
    <td>62.78</td>
    <td>55.83</td>
    <td>75.77</td>
    <td>78.67</td>
    <td>31.21</td>
    <td>40.40</td>
    <td>83.36</td>
    <td>62.45</td>
    <td>80.05</td>
    <td>52.90</td>
    <td>9.71</td>
    <td>18.16</td>
    <td>14.12</td>
    <td>11.05</td>

  </tr>
  <tr>
    <th>Ours</th>
    <td>61.67</td>
    <td>54.57</td>
    <td>61.32</td>
    <td>55.04</td>
    <td>76.48</td>
    <td>78.89</td>
    <td>29.74</td>
    <td>40.60</td>
    <td>83.24</td>
    <td>66.43</td>
    <td>79.76</td>
    <td>52.30</td>
    <td>9.98</td>
    <td>18.67</td>
    <td>14.39</td>
    <td>11.37</td>

  </tr>
  </tr>
    <th>Ours iters=1K,disable_quanted_input </th>
    <td>61.47</td> <! acc avg -->
    <td>55.41</td> <! MMLU -->
    <td>61.77</td> <! Lambada_openai -->
    <td>54.92</td> <! Hellsaswag -->
    <td>76.40</td> <! Winogrande -->
    <td>78.29</td> <! Piqa -->
    <td>31.09</td> <! Truthfulqa -->
    <td>40.0</td> <! Openbookqa -->
    <td>83.24</td> <! Boolq -->
    <td>63.54</td> <! RTE -->
    <td>79.29</td> <! Arc easy -->
    <td>52.22</td> <! Arc Challenge  -->
    <td>9.97</td>  <! wikitext2 ppl  -->
    <td>18.63</td> <! ptb_new ppl  -->
    <td>14.37</td>    <! c4_new ppl  -->
    <td>11.35</td> <! lm-eval wikitext ppl  -->
  </tr>
</br>

</table>  

### Other data W2G32
<table border="1">
  <tr>
    <th>Model</th>
    <th>Method </th>
    <th>Acc AVG.</th>
    <th>MMLU</th>
    <th>Lamb.</th>
    <th>Hella.</th>
    <th>Wino.</th>
    <th>Piqa</th>
    <th>Truth.</th>
    <th>Open.</th>
    <th>Boolq</th>
    <th>RTE</th>
    <th>ARC-e</th>
    <th>ARC-c.</th>
    <th>wikitext2 ppl
    <th>ptb_new ppl</th>
    <th>c4_new ppl</th>
    <th>lm_eval wikitext ppl</th>
   
  </tr>
  <tr>
    <td rowspan="3">mistralai/Mistral-7B </td>
    <th>FP16</th>
    <td>63.30 </td>
    <td>61.35 </td>
    <td>75.68 </td>
    <td>61.27 </td>
    <td>74.03 </td>
    <td>80.79 </td>
    <td>28.03 </td>
    <td>32.80 </td>
    <td>83.67 </td>
    <td>67.51 </td>
    <td>80.81 </td>
    <td>50.34 </td>
    <td>5.25 </td>
    <td>35.00 </td>
    <td>8.38 </td>
    <td>-</td>
  </tr>
  </tr>
    <th>Ours iters=1K </th>
    <td>56.44 </td>
    <td>47.38 </td>
    <td>67.26 </td>
    <td>55.06 </td>
    <td>67.88 </td>
    <td>77.75 </td>
    <td>26.19 </td>
    <td>26.40 </td>
    <td>78.07 </td>
    <td>58.12 </td>
    <td>74.20 </td>
    <td>42.49 </td>
    <td>7.14 </td>
    <td>56.78 </td>
    <td>10.71 </td>
    <td>-</td> <! lm-eval wikitext ppl  -->
  </tr>
  </tr>
    <th>Ours iters=4K,minmax_lr=0.0005  </th>
    <td>57.16 </td>
    <td>50.28 </td>
    <td>67.03 </td>
    <td>55.37 </td>
    <td>68.11 </td>
    <td>77.53 </td>
    <td>26.44 </td>
    <td>26.00 </td>
    <td>80.58 </td>
    <td>58.12 </td>
    <td>75.63 </td>
    <td>43.69 </td>
    <td>7.07 </td>
    <td>51.88 </td>
    <td>10.67 </td>
    <td>-</td> <! lm-eval wikitext ppl  -->
  </tr>

  <tr>
    <td rowspan="3">Meta/LLaMA-2-13B </td>
    <th>FP16</th>
    <td>61.42 </td>
    <td>52.86 </td>
    <td>76.77 </td>
    <td>60.04 </td>
    <td>72.14 </td>
    <td>79.05 </td>
    <td>25.95 </td>
    <td>35.20 </td>
    <td>80.55 </td>
    <td>65.34 </td>
    <td>79.38 </td>
    <td>48.38 </td>
    <td>4.88 </td>
    <td>50.93 </td>
    <td>6.73 </td>
    <td>7.90 </td>
 
  </tr>
<tr>
    <th>Ours iters=1K,minmax_lr=0.002</th>
    <td>56.95 </td>
    <td>42.39 </td>
    <td>70.87 </td>
    <td>55.15 </td>
    <td>68.03 </td>
    <td>77.37 </td>
    <td>24.11 </td>
    <td>30.80 </td>
    <td>77.58 </td>
    <td>64.62 </td>
    <td>75.63 </td>
    <td>39.93 </td>
    <td>6.26 </td>
    <td>78.83 </td>
    <td>8.70 </td>
    <td>11.25 </td>
  </tr>
  <tr>
    <th>Ours iters=2K,minmax_lr=0.001</th>
    <td>57.53 </td>
    <td>44.42 </td>
    <td>71.63 </td>
    <td>55.23 </td>
    <td>68.03 </td>
    <td>76.66 </td>
    <td>24.48 </td>
    <td>32.00 </td>
    <td>76.91 </td>
    <td>65.70 </td>
    <td>76.09 </td>
    <td>41.64 </td>
    <td>6.27 </td>
    <td>75.40 </td>
    <td>8.70 </td>
    <td>11.22 </td>
  </tr>

 <tr>
    <td rowspan="3">Meta/LLaMA-2-7B </td>
    <th>FP16</th>
    <td>57.98 </td>
    <td>42.69 </td>
    <td>73.90 </td>
    <td>57.15 </td>
    <td>68.90 </td>
    <td>78.07 </td>
    <td>25.21 </td>
    <td>31.40 </td>
    <td>77.74 </td>
    <td>62.82 </td>
    <td>76.35 </td>
    <td>43.52 </td>
    <td>5.47 </td>
    <td>37.92 </td>
    <td>7.26 </td>
    <td>8.79 </td>
  </tr>
  </tr>
    <th>Ours iters=1K,minmax_lr=0.002 </th>
    <td>52.29 </td>
    <td>27.14 </td>
    <td>65.48 </td>
    <td>50.25 </td>
    <td>66.61 </td>
    <td>74.54 </td>
    <td>24.11 </td>
    <td>29.80 </td>
    <td>73.30 </td>
    <td>56.68 </td>
    <td>70.20 </td>
    <td>37.12 </td>
    <td>8.72 </td>
    <td>1692.95 </td>
    <td>10.06 </td>
    <td>12.80 </td>
  </tr>
  </tr>
    <th>Ours iters=2K,minmax_lr=0.0005  </th>
    <td>52.32 </td>
    <td>28.26 </td>
    <td>64.16 </td>
    <td>50.66 </td>
    <td>64.80 </td>
    <td>75.14 </td>
    <td>23.87 </td>
    <td>30.20 </td>
    <td>71.74 </td>
    <td>57.76 </td>
    <td>71.13 </td>
    <td>37.80 </td>
    <td>8.54 </td>
    <td>0.00 </td>
    <td>10.14 </td>
    <td>0.00 </td>
  </tr>

</table>
</table>
