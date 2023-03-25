# A CLustered TPP instances are given in file name: Clustered-Travelling-Purchaser-Problem_INSTANCES
# The instance Clustered-Travelling-Purchaser-Problem_INSTANCES contains 25 standard TSP problems with Product costs (p_cost_file_INSTANCES NAME) and availability (avl_file_INSTANCES NAME) generated according to Angelelli et al. 2017.
# Here is a python code included in the instances file Clustered-Travelling-Purchaser-Problem_INSTANCES with the file name: VLCGA_TPP_instances(Corrected purchasing cost calculated at line number 1113 ) a reuslt file also included with file name:result.xlsx; the code contains specific information like instance generations, k-means, VLCGA and results updations are given.
# A obtained results using k-menas-VLCGA are given in file(xlsx) name:Clustered_TPP_Result
# A file name:CPLEX_TPP contains solutions of the standard TPP using CPLEX by OPL with two names tpp_mod and TPP_Test; TPP_Test works for standard TPP instances given in Clustered-Travelling-Purchaser-Problem_INSTANCES as well as in given file: CapEuclideo.50.50.1.1_Test.dat; Other OPL file tpp_mod works for multi-vehicle clustered TPP with multi items and different demand setting, for this purpose a data-set.txt file is included. The information for various demands, availability and prices are included in data-set.txt
# A clustered TPP file name: Multi_Vehicle_Clustered-TPP for multiple items is given the data set, and a python code, bays29.txt contain distance matrix, availability by avl_multi_file.txt, product cots by p_cost_multi_file; A python code for multi-vehicle clustered TPP:Multi-vehicle_Clustered_TPP_k-means-VLCGA.py
# A clustered TPP file name: Multi_Vehicle_Clustered-TPP-Single-item for single items is given the data set and a python code:MVCLuTPP_Scenario-1.py , MVCLuTPP_Scenario-2.py
# A file name: Multi_Vehicle_Clustered-TPP_Random contain a file: CTPP-stocastic.py for stochastic instances.
#A clean version of code is given in folder name:CLuTPP_final_version, which contains four sub-folder nmae:CPLEX_TPP_V2, Name:Multi_Vehicle_Clustered_TPP_Single_item_V2, Name:Multi_Vehicle_Clustered-TPP_Random_V2 and name: Multi_Vehicle_Clustered-TPP_V2.
Here sub-folder name:CPLEX_TPP_V2 contains seven data set namely:data-set, contains only given data, file name:multi_cl-1 contains data for single cluster  using multi item, multi_cl-2 contains data for two cluster and similarly file name: multi_cl-3 for multi item. File name:sinlge_cl-1, sinlge_cl-2, sinlge_cl-3 for single item using cluster 1, 2 and 3 respectively. File name:tpp_mod contains CPLEX models. To execute this code run only file name:tpp_mod, and change the dataset as per given in the folder. 
# The sub-folder name:Multi_Vehicle_Clustered_TPP_Single_item_V2 under the folder name:CLuTPP_final_version contains seven files namely:avl_file, bays29.tsp, bays29, MVCLuTPP_Scenario-1.py, MVCLuTPP_Scenario-2.py and 
# The sub-folder name:Multi_Vehicle_Clustered_TPP_Single_item_V2 under the folder name:CLuTPP_final_version contains seven files namely:avl_file, bays29.tsp, bays29, MVCLuTPP_Scenario-1.py, MVCLuTPP_Scenario-2.py and p_cost_file respectively. Here python code for variable length genetic algorithm with k-menas clustering and a relinking methods.  All data for single item, Some time check the seeds number and the crossover mechanism makes infeasible solutions, you may stuck and start with new seeds. 
# The sub-folder name:Multi_Vehicle_Clustered-TPP_V2 under the folder name:CLuTPP_final_version contains five files namely:CTPP-multi_cnt_to_depo_multi_vehicle.py contains the python code for multiple items. Here python code for variable length genetic algorithm with k-means clustering and a relinking methods.  All data for multi item, Some time check the seeds number and the crossover mechanism makes infeasible solutions, you may stuck and start with new seeds. 
# The sub-folder name:Multi_Vehicle_Clustered-TPP_Random_V2 under the folder name:CLuTPP_final_version contains five files namely:CTPP-stocastic.py contains the python code for multiple items under random phenomena. Here python code for variable length genetic algorithm with k-means clustering and a relinking methods.  All data for multi item, Some time check the seeds number and the crossover mechanism makes infeasible solutions, you may stuck and start with new seeds.
  
