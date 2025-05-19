# setting CRAN mirror
options(repos = c(CRAN = "https://cran.rstudio.com/"))

# START : Dependency Check
requiredPackage <- c("BiocManager", "biomartr")
requiredPackage_BC <- c("biomaRt","Biostrings")

for( p in requiredPackage ){
  if( !(p %in% installed.packages())){
    install.packages( p )
  }
}

library(BiocManager)

for( p in requiredPackage_BC ){
  if( !( p %in% installed.packages( ))){
    BiocManager::install( p )
  }
}


# EDN : Dependency Check

# START : Library Installation
library( biomartr )

# END : Library Installation

# START : Downloading genome information for each species
assembly_accession <- c("GCF_000001405.39") #GRCh38(2013/12/17)
for( acc in assembly_accession ){
  # Setting the directory to store the genome data
     db <- "refseq"
     save_dir <- paste("./" , acc , sep = "")
  
  ## If one of the genome specified in assembly_accession is downloaded, following code will be ignored omitting the exaustive time to download.
  if( !( dir.exists( save_dir ))){
    ## Following will be replaced to biomartr::getCollection function.
    ## Currently the code of biomartr::getCollection may contain bug and I have already reported it to developer
    ## ISSUE : https://github.com/ropensci/biomartr/issues/53
    biomartr::getGenome( db = db , organism = acc , path = file.path( save_dir , "genomes"), reference = FALSE) ## Scaffold or Contig Download
    biomartr::getCDS(  db = db , organism = acc , path = file.path( save_dir , "CDSs") ) # CDS Download
    biomartr::getRNA( db = db , organism = acc , path = file.path( save_dir , "RNA")) # RNA Download
    biomartr::getProteome( db = db , organism = acc , path = file.path( save_dir , "proteomes")) # Protein sequence Download
    biomartr::getGFF( db = db , organism = acc , path = file.path( save_dir , "annotations")) # GFF Download
    # biomartr::getAssemblyStats( db = db , organism = acc , path = file.path( save_dir , "genomeassembly_stats")) # Meta data Download
  }
  else{
    print( paste( save_dir , "already exists!"))
  }
}
