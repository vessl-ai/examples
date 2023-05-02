---
description: Dashboard for payment information
---

# Billing Information

You can check your payment information and credit balance on the billing page.&#x20;

![](<../../../.gitbook/assets/image (222).png>)

### How is usage calculated?

VESSL charges based on compute, storage, and network usage. Check the following pricing table for each resource type.&#x20;

#### ap-northeast-2

| Resource Name          | CPU(Core) | RAM(Gi) | GPU              | Spot instance | Price($/hour) |
| ---------------------- | --------- | ------- | ---------------- | ------------- | ------------- |
| v1.cpu-0.mem-1         | 0.25      | 1       | X                | X             | 0             |
| v1.cpu-2.mem-6         | 2         | 6       | X                | X             | 0.118         |
| v1.cpu-2.mem-6.spot    | 2         | 6       | X                | O             | 0.0315        |
| v1.cpu-4.mem-13        | 4         | 13      | X                | X             | 0.236         |
| v1.cpu-4.mem-13.spot   | 4         | 13      | X                | O             | 0.0629        |
| v1.t4-1.mem-13         | 4         | 13      | T4\*1            | X             | 0.647         |
| v1.t4-1.mem-13.spot    | 4         | 13      | T4\*1            | O             | 0.1941        |
| v1.t4-1.mem-54         | 16        | 54      | T4\*1            | X             | 1.481         |
| v1.t4-1.mem-54.spot    | 16        | 54      | T4\*1            | O             | 0.4443        |
| v1.t4-4.mem-163        | 48        | 163     | T4\*4            | X             | 4.812         |
| v1.t4-4.mem-163.spot   | 48        | 163     | T4\*4            | O             | 1.4436        |
| v1.k80-1.mem-52        | 4         | 52      | K80 \* 1         | X             | 1.465         |
| v1.k80-1.mem-52.spot   | 4         | 52      | K80 \* 1         | O             | 0.4395        |
| v1.k80-8.mem-480       | 32        | 480     | K80 \* 8         | X             | 11.72         |
| v1.k80-8.mem-480.spot  | 32        | 480     | K80 \* 8         | O             | 3.516         |
| v1.k80-16.mem-724      | 64        | 724     | K80 \* 16        | X             | 23.44         |
| v1.k80-16.mem-724.spot | 64        | 724     | K80 \* 16        | O             | 7.032         |
| v1.v100-1.mem-52       | 8         | 52      | V100 (16GB) \* 1 | X             | 4.234         |
| v1.v100-1.mem-52.spot  | 8         | 52      | V100 (16GB) \* 1 | O             | 1.2702        |
| v1.v100-4.mem-232      | 32        | 232     | V100 (16GB) \* 4 | X             | 16.936        |
| v1.v100-4.mem-232.spot | 32        | 232     | V100 (16GB) \* 4 | O             | 5.0808        |
| v1.v100-8.mem-460      | 64        | 460     | V100 (16GB) \* 8 | X             | 33.872        |
| v1.v100-8.mem-460.spot | 64        | 460     | V100 (16GB) \* 8 | O             | 10.1616       |

#### us-west-2

| Resource Name           | CPU(Core) | RAM(Gi) | GPU              | Spot instance | Price($/hour) |
| ----------------------- | --------- | ------- | ---------------- | ------------- | ------------- |
| v1.cpu-0.mem-1          | 0.25      | 1       | X                | X             | 0             |
| v1.cpu-2.mem-6          | 2         | 6       | X                | X             | 0.096         |
| v1.cpu-2.mem-6.spot     | 2         | 6       | X                | O             | 0.0338        |
| v1.cpu-4.mem-13         | 4         | 13      | X                | X             | 0.192         |
| v1.cpu-4.mem-13.spot    | 4         | 13      | X                | O             | 0.077         |
| v1.t4-1.mem-13          | 4         | 13      | T4\*1            | X             | 0.526         |
| v1.t4-1.mem-13.spot     | 4         | 13      | T4\*1            | O             | 0.1578        |
| v1.t4-1.mem-54          | 16        | 54      | T4\*1            | X             | 1.204         |
| v1.t4-1.mem-54.spot     | 16        | 54      | T4\*1            | O             | 0.3612        |
| v1.t4-4.mem-163         | 48        | 163     | T4\*4            | X             | 3.912         |
| v1.t4-4.mem-163.spot    | 48        | 163     | T4\*4            | O             | 1.2719        |
| v1.k80-1.mem-52         | 4         | 52      | K80 \* 1         | X             | 0.9           |
| v1.k80-1.mem-52.spot    | 4         | 52      | K80 \* 1         | O             | 0.27          |
| v1.k80-8.mem-480        | 32        | 480     | K80 \* 8         | X             | 7.2           |
| v1.k80-8.mem-480.spot   | 32        | 480     | K80 \* 8         | O             | 2.16          |
| v1.k80-16.mem-724       | 64        | 724     | K80 \* 16        | X             | 14.4          |
| v1.k80-16.mem-724.spot  | 64        | 724     | K80 \* 16        | O             | 4.32          |
| v1.v100-1.mem-52        | 8         | 52      | V100 (16GB) \* 1 | X             | 3.06          |
| v1.v100-1.mem-52.spot   | 8         | 52      | V100 (16GB) \* 1 | O             | 0.918         |
| v1.v100-4.mem-232       | 32        | 232     | V100 (16GB) \* 4 | X             | 12.24         |
| v1.v100-4.mem-232.spot  | 32        | 232     | V100 (16GB) \* 4 | O             | 3.672         |
| v1.v100-8.mem-752       | 96        | 752     | V100 (32GB) \* 8 | X             | 31.212        |
| v1.v100-8.mem-752.spot  | 96        | 480     | V100 (32GB) \* 8 | O             | 9.3636        |
| v1.a100-8.mem-1140.spot | 96        | 1140    | A100 \* 8        | O             | 9.8318        |



###
