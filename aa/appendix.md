# Appendix


## Code snippets
```
### R code to generate exploratory data analysis report using DataExplorer package
df %>%
    create_report(
        output_file = paste("Report", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"), sep=" - "),
        report_title = "EDA Report - Corporate Credit Rating",
        y = "Rating"
    )
```