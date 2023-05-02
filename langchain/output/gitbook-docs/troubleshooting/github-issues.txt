# GitHub Issues

If you're having trouble connecting to GitHub, try troubleshooting on GitHub, not on VESSL Web Console. You may encounter the following issues and errors while creating a project on VESSL:

* GitHub redirection issue
* Invalid Github repository error
* Unauthorized GitHub erorr
* GitHub connection error
* GitHub rate limit error

### GitHub redirection issue

Some users who share your organization may find that clicking **New Project** button on the VESSL Web Console redirects to GiHub Apps. In this case, you can update the **Repository access** and activate **Save** button on the GitHub Apps page.&#x20;

![](<../.gitbook/assets/image (152).png>)

### Invalid GitHub Repository Error

The invalid GitHub repository error usually occurs when the repository does not exist or the VESSL GitHub App is not installed. Check whether the GitHub repository exists and make sure VESSL GitHub App is insatlled on your GitHub account. You can verify the installation under&#x20;

* Personal Settings -> Applications -> Authorized GitHub Apps, or
* Organization settings -> Installed GitHub Apps

![](<../.gitbook/assets/image (90).png>)

![](<../.gitbook/assets/image (136).png>)

### Unauthorized GitHub Error

The unauthorized GitHub Error may occur when the GitHub authorization expires. Resolve the error by following the steps below:

1. Uninstall the VESSL GitHub App on your GitHub account under Settings -> Applications.
2. Click **New Project** on the organization page to reinstall the VESSL GitHub App.

![](<../.gitbook/assets/image (88).png>)

![](<../.gitbook/assets/image (128).png>)

### GitHub Connection Error

The GitHub connection error occurs when the GitHub server is temporarily unavailable.&#x20;

### GitHub Rate Limit Error

According to[ GitHub Docs](https://docs.github.com/en/free-pro-team@latest/developers/apps/rate-limits-for-github-apps#:\~:text=Organization%20installations%20with%20more%20than,is%2012%2C500%20requests%20per%20hour.), the maximum rate limit for an installation is 12,500 requests per hour. If the GitHub Rate Limit Error occurs, try again later.
