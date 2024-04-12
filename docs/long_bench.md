# More results on LongBench

From the results, for vanilla multihead attention models, we recommend using KIVI-2, which can maintain the performance of the full-precision model while offering the best efficiency. For multiquery attention or group query attention, since the keys and values are already compressed, we recommend using KIVI-4, which can maintain the performance of the full-precision model in these cases.

### Table 1: Performance of LongChat-7b-v1.5-32K

The results of LongChat-7b-v1.5-32K on 15 tasks from LongBench. The model has 32K context length. We use a 32 group size and 128 residual length for both KIVI-2 and KIVI-4. The baseline is of full precision.

| Task             | LongChat-7b-v1.5-32K | w./ KIVI-2 | w./ KIVI-4 |
|------------------|:--------------------:|:------------:|:-------------:|
| NarrativeQA      | 20.65                | 20.79        | 20.49         |
| Qasper           | 29.42                | 28.69        | 28.90         |
| MultiFieldQA     | 43.15                | 41.02        | 43.24         |
| HotpotQA         | 33.05                | 32.91        | 33.07         |
| MuSiQue          | 14.66                | 13.82        | 14.66         |
| 2WikiMultihopQA  | 24.14                | 23.00        | 24.86         |
| GovReport        | 30.85                | 30.47        | 31.40         |
| QMSum            | 22.84                | 22.59        | 22.84         |
| MultiNews        | 26.55                | 26.28        | 26.52         |
| LCC              | 54.83                | 54.11        | 54.06         |
| RepoBench-P      | 58.94                | 57.62        | 58.77         |
| TriviaQA         | 83.99                | 83.19        | 83.88         |
| SAMSum           | 40.75                | 41.28        | 40.62         |
| TRec             | 66.50                | 66.50        | 67.00         |
| PassageRetrieval | 30.50                | 32.25        | 31.50         |
| **Average**      | **38.72**            | **38.30**    | **38.79**     |

### Table 2: Performance of Mistral-7B-Instruct-v0.2

The results of Mistral-7B-Instruct-v0.2 on 15 tasks from LongBench. The model has 32K context length and applies group query attention, which uses 8 heads for KV Cache, as opposed to the full 32 heads. We use a 32 group size and 128 residual length for both KIVI-2 and KIVI-4. The baseline is of full precision.

| Task             | Mistral-7B-Instruct-v0.2 | w./ KIVI-2 | w./ KIVI-4 |
|------------------|:------------------------:|:------------:|:-------------:|
| NarrativeQA      | 21.02                    | 20.61        | 20.97         |
| Qasper           | 29.41                    | 28.73        | 29.41         |
| MultiFieldQA     | 47.13                    | 44.88        | 46.52         |
| HotpotQA         | 36.53                    | 35.47        | 36.25         |
| MuSiQue          | 19.13                    | 17.95        | 19.53         |
| 2WikiMultihopQA  | 21.76                    | 20.68        | 21.66         |
| GovReport        | 32.59                    | 32.55        | 32.97         |
| QMSum            | 23.99                    | 23.65        | 24.06         |
| MultiNews        | 27.09                    | 26.54        | 26.89         |
| LCC              | 53.49                    | 53.03        | 53.33         |
| RepoBench-P      | 51.40                    | 51.16        | 51.41         |
| TriviaQA         | 86.23                    | 86.00        | 86.23         |
| SAMSum           | 43.04                    | 43.34        | 43.34         |
| TRec             | 71.00                    | 71.00        | 71.00         |
| PassageRetrieval | 89.33                    | 80.83        | 89.42         |
| **Average**      | **43.54**                | **42.43**    | **43.53**     |
