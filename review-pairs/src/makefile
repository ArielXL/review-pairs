.PHONY: clean

NAME 	   	:= Classify the pairs
VERSION		:= 1.0
DEVELOPERS	:= Ariel Plasencia Díaz
COPYRIGHT  	:= Copyright © 2025: $(DEVELOPERS)


run_make_pairs: ## Run the make pairs
	python3 make_pairs.py

run_review_pairs: ## Run the review pairs app
	streamlit run review_pairs.py

info: ## Display project description
	@echo "$(NAME) v$(VERSION)"
	@echo "$(COPYRIGHT)"

version: ## Show the project version
	@echo "$(NAME) v$(VERSION)"

clean: ## Remove temporary files
	rm -rf ../pairs.xlsx

install: ## Install the project dependencies
	pip3 install -r ../requirements.txt

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'