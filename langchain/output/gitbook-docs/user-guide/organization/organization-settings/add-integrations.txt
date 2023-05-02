# Add Integrations

### Integrating your service to VESSL

You can integrate various services to VESSL using AWS, Docker, and SSH Keys. The integrated AWS and Docker credentials are used to manage private docker images, whereas the SSH Keys are used for authorized keys for SSH connection.

### AWS Credentials&#x20;

To integrate your AWS account, you need an AWS access key associated with your [IAM](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html). You can create this key by following this [guide from AWS](https://docs.aws.amazon.com/IAM/latest/UserGuide/id\_credentials\_access-keys.html). Once you have your key, click **ADD INTEGRATION** and fill in the form.&#x20;

![](<../../../.gitbook/assets/image (14).png>)

You can integrate multiple AWS credentials to your organization. You can also revoke your credentials by simply clicking the trash button on the right.&#x20;

![](<../../../.gitbook/assets/image (16).png>)

{% hint style="info" %}
If you want to pull images from [ECR](https://docs.aws.amazon.com/AmazonECR/latest/userguide/what-is-ecr.html) , make sure to provide the [ECR pull policy granted account.](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-policy-examples.html)
{% endhint %}

### Docker Credentials

To integrate your Docker account, click **ADD INTEGRATION** and fill in your Docker credentials.

![](<../../../.gitbook/assets/image (17).png>)

### GitHub

To integrate your GitHub account, click ADD INTEGRATION. Grant repository access to VESSL App in the repository access section and click save in GitHub.

![](<../../../.gitbook/assets/image (220).png>)
