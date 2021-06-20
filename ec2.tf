provider "aws" {
  region                  = "ap-southeast-1"
  profile                 = "default"
}

#-------------AWS_INSTANCE--------------------------
resource "aws_instance" "myOS" {
  ami           = "ami-0e8e39877665a7c92"
  instance_type = "t2.micro"
  key_name = "task38"
 tags = {
    Name = "Os by face_recognition"	
  }
}

#------------AWS VOLUME------------------------
resource "aws_ebs_volume" "vol" {
  availability_zone = aws_instance.myOS.availability_zone
  size              = 1

  tags = {
    Name = "HD by face recognition"
  }
}

#---------------ATTACH VOLUME------------------
resource "aws_volume_attachment" "ebs_att" {
  device_name = "/dev/sdc"
  volume_id   = aws_ebs_volume.vol.id
  instance_id = aws_instance.myOS.id
  force_detach = true
}

resource "null_resource"  "nullremote" {
 connection {
    type     = "ssh"
    user     = "ec2-user"
    private_key = file("./task38.pem")
    host     = aws_instance.myOS.public_ip
  }

provisioner "remote-exec" {
    inline = [
      "sudo mkfs.ext4 /dev/xvdc",
      "sudo  mount /dev/xvdc  /",
    ]
  }
}


