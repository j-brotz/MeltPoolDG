# How to document the change of parameters

If you modify any of the input parameters of MeltPoolDG, please add an entry into the 
parameters-changes.md file based on the following template. Try to stick to a max. number of 100 
characters per line.

.
.
.
----------------------------------------------------------------------------------------------------
<change type>:
<changes>

<description>

(<author name>, <date>)
----------------------------------------------------------------------------------------------------
<changes>
.
.
.

Consider the following style conventions:

 - <change type>: select the change type if a parameter is Added/Removed/Renamed/Other; other can be
                  used for a change of the default value
 - <changes>: use the git diff syntax to describe the changes in the *.json tree of the input 
              parameters: 
              * A line starting with "+" indicates that a parameter is added.
              * A line starting with "-" indicates that a parameter is removed.
              * A combination of a line starting with "+" and a line starting with "-" can be used to
              indicate a renaming or a change in the default value;
 - <description>: (optional) short description of the changes 
 - <author name>: first name and last name
 - <date>: note the date of your changes in format YYYY-MM-DD
