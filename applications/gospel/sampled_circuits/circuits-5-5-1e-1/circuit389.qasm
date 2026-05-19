OPENQASM 2.0;
include "qelib1.inc";
qreg q390[5];
rx(5*pi/4) q390[3];
cx q390[3],q390[2];
cx q390[2],q390[1];
cx q390[3],q390[4];
cx q390[1],q390[0];
