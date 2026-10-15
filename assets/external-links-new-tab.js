var links = document.links;

for (var i = 0; i < links.length; i++) {
  if (links[i].hostname == 'localhost' || links[i].hostname == '127.0.0.1') {
    continue;
  }

  if (!links[i].hostname.includes("pytorch.kr") && !links[i].hostname.includes("surge.sh")) {
    links[i].target = '_blank';
  }
}