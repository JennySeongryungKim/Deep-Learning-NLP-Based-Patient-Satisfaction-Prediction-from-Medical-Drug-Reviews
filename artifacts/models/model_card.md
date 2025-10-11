
# Model Card: WebMD Drug Review Sentiment Analysis

## Model Performance


### TEXTCNN
- accuracy: 0.6697
- macro_f1: 0.6175
- cohen_kappa: 0.4900

### IMPROVED_BILSTM
- accuracy: 0.7230
- macro_f1: 0.6251
- cohen_kappa: 0.5428

### IMPROVED_BERT
- accuracy: 0.7912
- macro_f1: 0.6723
- cohen_kappa: 0.6469


## Dataset Information
- Source: WebMD Drug Reviews
- Task: 10-class satisfaction prediction
- Preprocessing: Text cleaning, negation tagging, medical NER

## Limitations
- May exhibit bias across demographic subgroups
- Performance varies by drug type and condition
- Negation handling may miss complex linguistic patterns

## Ethical Considerations
- Should not be used as sole medical decision-making tool
- Requires human oversight for critical applications
- Monitor for fairness across protected attributes

## Usage Guidelines
- Intended for research and analysis purposes
- Regular monitoring for data drift recommended
- Consider ensemble approaches for production use
