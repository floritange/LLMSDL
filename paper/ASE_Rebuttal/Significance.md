# #139 Rebuttal

## Results of Mann-Whitney U-test:

*Default Model: Qwen2.5*

- *** p < 0.001 (highly significant)
- ** p < 0.01 (significant)
- \* p < 0.05 (slightly significant)
- ns: Not significant

**Key Findings:**
1. **Performance Comparison**: LLMSDL significantly outperforms baseline methods across all evaluation metrics, with most results reaching high statistical significance (p < 0.001)
2. **Cost Analysis**: Compared to SWE-agent, LLMSDL significantly reduces computational costs; compared to lightweight methods (Agentless/AutoCodeRover), LLMSDL has increased costs but achieves more substantial performance improvements
3. **Statistical Reliability**: All performance improvements pass statistical significance tests, demonstrating the effectiveness and reliability of the LLMSDL method

### Table: Effectiveness Comparison - Statistical Significance Analysis

| Comparison | HR@k | HR@k+2 | HR@k+4 | NDCG@k | NDCG@k+2 | NDCG@k+4 |
|------------|------|--------|--------|--------|----------|----------|
| **LLMSDL vs SWE-agent** | 58.3% vs 48.7%*** | 72.8% vs 54.8%*** | 80.1% vs 55.6%*** | 50.4% vs 39.7%*** | 58.8% vs 43.4%*** | 62.2% vs 43.7%*** |
| **LLMSDL vs Agentless** | 58.3% vs 43.9%*** | 72.8% vs 53.1%*** | 80.1% vs 60.3%*** | 50.4% vs 37.7%*** | 58.8% vs 44.2%*** | 62.2% vs 47.4%*** |
| **LLMSDL vs AutoCodeRover** | 58.3% vs 49.3%** | 72.8% vs 63.4%*** | 80.1% vs 67.4%*** | 50.4% vs 42.1%** | 58.8% vs 49.8%*** | 62.2% vs 51.4%*** |
| **LLMSDL vs Agentless\*** | 58.3% vs 44.2%*** | 72.8% vs 53.3%*** | 80.1% vs 60.9%*** | 50.4% vs 37.5%*** | 58.8% vs 44.5%*** | 62.2% vs 47.8%*** |
| **LLMSDL vs AutoCodeRover\*** | 58.3% vs 49.0%** | 72.8% vs 62.4%*** | 80.1% vs 66.4%*** | 50.4% vs 41.8%*** | 58.8% vs 49.5%*** | 62.2% vs 51.1%*** |

### Table: Average Costs Comparison - Statistical Significance Analysis

| Comparison | Input Tokens | Output Tokens |
|------------|--------------|---------------|
| **LLMSDL vs SWE-agent** | 19,466 vs 270,112*** | 955 vs 5,420*** |
| **LLMSDL vs Agentless** | 19,466 vs 1,931*** | 955 vs 505*** |
| **LLMSDL vs AutoCodeRover** | 19,466 vs 2,690*** | 955 vs 905*** |
| **LLMSDL vs Agentless\*** | 19,466 vs 1,906*** | 955 vs 513*** |
| **LLMSDL vs AutoCodeRover\*** | 19,466 vs 2,737*** | 955 vs 904*** |