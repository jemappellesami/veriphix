OPENQASM 2.0;
include "qelib1.inc";
qreg q289[6];
cx q289[5],q289[4];
cx q289[3],q289[4];
cx q289[2],q289[3];
cx q289[2],q289[1];
cx q289[1],q289[0];
