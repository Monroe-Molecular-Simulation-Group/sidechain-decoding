
file=$1
curl -X POST -H "Content-Type: application/json" -d @$file https://search.rcsb.org/rcsbsearch/v2/query #>> results.json

