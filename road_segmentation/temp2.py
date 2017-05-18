if im_patch.shape[0] < 2*cf+h and im_patch.shape[1] == 2*cf+w:
     pad_size = 2*cf+h - im_patch.shape[0]
     im_patch = numpy.pad(im_patch, ((0,pad_size),(0,0) ,(0,0)), 'constant')
     title = 'i,j = ' + str(i) + ', ' + str(j)
     plt.title(title)
     plt.imshow(im_patch)
     plt.show()
 elif im_patch.shape[1] < 2*cf+w and im_patch.shape[0] == 2*cf+h:
     pad_size = 2*cf+w - im_patch.shape[1]
     im_patch = numpy.pad(im_patch, ((0,0),(0,pad_size), (0,0)), 'constant')
     title = 'i,j = ' + str(i) + ', ' + str(j)
     plt.title(title)
     plt.imshow(im_patch)
     plt.show()
 elif im_patch.shape[1] < 2*cf+w and im_patch.shape[0] < 2*cf+h:
     pad_size0 = 2*cf+h - im_patch.shape[0]
     pad_size1 = 2*cf+w - im_patch.shape[1]
     im_patch = numpy.pad(im_patch, (( 0,pad_size0),(0,pad_size1),(0,0)), 'constant')
     title = 'i,j = ' + str(i) + ', ' + str(j)
     plt.title(title)
     plt.imshow(im_patch)
     plt.show()