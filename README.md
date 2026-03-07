README file for Thesis Project





|thesis\_project/



   |artifacts/

        |checkpoints/

        |datasets/

             |clean/

 		 |clean\_dataset\_v1\_seed8.mat

 		 |clean\_dataset\_v1\_seed49.mat

 		 |clean\_dataset\_v1\_seed888.mat

             |impaired/

 		 |impaired\_dataset\_v1\_seed8\_eval.mat

 		 |impaired\_dataset\_v1\_seed8\_train.mat

 		 |impaired\_dataset\_v1\_seed49\_train.mat

 		 |impaired\_dataset\_v1\_seed49\_eval.mat

 		 |impaired\_dataset\_v1\_seed888\_train.mat

 		 |impaired\_dataset\_v1\_seed888\_eval.mat

        |figs/

        |logs/

 

   |configs/



   |contracts/

 	|matlab\_python

             |Matlab\_Python\_Interface\_Contract.md



   |env/



   |matlab/

        |+clean/

 	     |generate\_clean\_dataset.m

 	     |generate\_clean\_sample.m

 	     |generate\_sample\_params.m

 	     |generate\_stat\_report\_clean.m

 	     |get\_active\_fields.m

 	     |init\_clean\_param\_record.m

             |report\_clean\_dataset\_v1.m

 	     |set\_sample\_rng.m

  	     |synthesize\_clean\_signal\_class0.m

  	     |synthesize\_clean\_signal\_class1.m

  	     |synthesize\_clean\_signal\_class2.m

  	     |synthesize\_clean\_signal\_class3.m

  	     |synthesize\_clean\_signal\_class4.m

  	     |synthesize\_clean\_signal\_class5.m

  	     |synthesize\_clean\_signal\_class6.m

 	     |validate\_active\_fields.m

        |+core/

 	     |compute\_artifact\_hash.m

 	     |get\_canonical\_spec.m

 	     |validate\_spec\_structure.m

        |+impaired/

 	     |apply\_impairment.m

 	     |generate\_impaired\_dataset.m

 	     |generate\_stat\_report\_impaired.m

 	     |init\_imp\_param\_record.m

 	     |report\_impaired\_dataset\_v1.m

        |export/

        |tests/



   |python/

        |notebooks/

        |src/

            |dataio/

 		|\_\_init\_\_.py

 		|contract.py

 		|dataset\_artifact.py

 		|exceptions.py

 		|loader.py

            |eval/

            |models/

 	    |preprocessing

 		|\_\_init\_\_.py

 		|dataset\_builder.py

 		|splitting.py

 		|stft.py

            |train/

            |utils/

 	    |validation

 		|\_\_init\_\_.py

 		|baseline.py

 		|checks.py

 		|exceptions.py

 		|features.py

 		|repro.py

 		|runner.py

 		|stats.py

 		|summary.py

 		|types.py



        |tests/

 	|\_\_init\_\_.py ...(nothing written here)



   |reports/

        |architectural/

             |Artifact Report on Clean Dataset Generator.md

             |Artifact Report on Impaired Dataset Generator.md

        |experiments/

        |statistical/

 	     |clean\_dataset\_v1\_seed8\_statisticl\_report.md

 	     |clean\_dataset\_v1\_seed49\_statisticl\_report.md

 	     |clean\_dataset\_v1\_seed888\_statisticl\_report.md

 	     |impaired\_dataset\_v1\_seed8\_eval\_report.md

 	     |impaired\_dataset\_v1\_seed8\_train\_report.md

 	     |impaired\_dataset\_v1\_seed49\_eval\_report.md

 	     |impaired\_dataset\_v1\_seed49\_train\_report.md

 	     |impaired\_dataset\_v1\_seed888\_eval\_report.md

 	     |impaired\_dataset\_v1\_seed888\_train\_report.md

 	     |validation\_seed49.json



        |README.md



   |scripts/

 	|pipe.py 

 	|run\_clean\_pipeline.m

 	|run\_impaired\_pipeline.m

 	|run\_validation.py

   |specs/

 	|dataset\_spec\_v1.md

 	|signal\_spec\_v1.md

 	|system\_spec\_v1.md



   |README.md \[this folder placement is stored here]

