OPENQASM 2.0;
include "qelib1.inc";
qreg q75[7];
cx q75[5],q75[4];
cx q75[3],q75[4];
cx q75[2],q75[3];
cx q75[1],q75[2];
cx q75[1],q75[0];
