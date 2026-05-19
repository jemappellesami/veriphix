OPENQASM 2.0;
include "qelib1.inc";
qreg q681[5];
cx q681[4],q681[3];
rx(3*pi/2) q681[3];
cx q681[2],q681[3];
cx q681[1],q681[2];
cx q681[0],q681[1];
