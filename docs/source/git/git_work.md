# Git work with Github

## Set up git software on your OS 

+ download git for windows and then install

## Initalizae the enviroment

+ select a position for your work, such as `c:\\work_place`, then enter that path.

+ initialize the git enviroment  

    > `git init`

    > `git config --global user.name "9527atct"`  

    > `git config --global user.email "lqj_614_19@163.com"`  
    

+ pull the remote repository  
    > `git pull https://github.com/9527atct/9527atct.github.io.git`

## Do some works within the reposiory

+ edit the files within reposiory, that is, to add or edit or delete some files.

## Update the remote reposiory

+ update  
    > `git add .`

    > `git commit -m "commit information made by your self"`

    > `git push origin master`  

+ first update, need to configure the remote information  
    > `git remote add origin https://github.com/9527atct/9527atct.github.io.git`   
    
    > `git remote -v`